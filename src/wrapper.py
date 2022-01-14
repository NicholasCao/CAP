# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# type
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedModel
# config
from transformers import BertConfig, RobertaConfig, ElectraConfig, DebertaV2Config, AlbertConfig
# tokenizer
from transformers import BertTokenizer, RobertaTokenizer, ElectraTokenizer, DebertaV2Tokenizer, AlbertTokenizer
# mlm
from src.mlm_plus import BertForMaskedLMPlus, RobertaForMaskedLMPlus, AlbertForMaskedLMPlus
# nsp
from transformers import BertForNextSentencePrediction
# rtd
from transformers import ElectraForPreTraining
from src.deberta import DebertaV3ForRTD

from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange

from data_utils.utils import InputExample, METRICS, DEFAULT_METRICS
from src.config import WrapperConfig
from src.utils import get_embeddings, simple_accuracy, exact_match, compute_avg_metric
from src.preprocessor import Preprocessor, PREPROCESSORS
from src.task_helpers import TASK_HELPERS
from src.prompt_encoder import PromptEncoder
from src.prompt_decoder import MODELS

import log

logger = log.get_logger('root')


MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'mlm': BertForMaskedLMPlus,
        'nsp': BertForNextSentencePrediction
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'mlm': RobertaForMaskedLMPlus
    },
    'electra': {
        'config': ElectraConfig,
        'tokenizer': ElectraTokenizer,
        'rtd': ElectraForPreTraining
    },
    'deberta': {
        'config': DebertaV2Config, # DeBertaV3 use DebertaV2Config in transfomers
        'tokenizer': DebertaV2Tokenizer, # DeBertaV3 use DebertaV2Tokenizer in transfomers
        'rtd': DebertaV3ForRTD
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'mlm': AlbertForMaskedLMPlus
    }
}


class TransformerModelWrapper:
    """A wrapper around a Transformer-based language model."""

    def __init__(self, config: WrapperConfig):
        self.config = config

        # TODO
        # config_class: PretrainedConfig = MODEL_CLASSES[config.model_type]['config']
        tokenizer_class: PreTrainedTokenizer = MODEL_CLASSES[config.model_type]['tokenizer']
        model_class: PreTrainedModel = MODEL_CLASSES[config.model_type][config.wrapper_type]

        # self.base_model_config = config_class.from_pretrained(config.model_name_or_path)
        self.tokenizer = tokenizer_class.from_pretrained(config.model_name_or_path)
        self.base_model = model_class.from_pretrained(config.model_name_or_path)


        self.model: nn.Module = MODELS[config.wrapper_type](self.base_model, num_labels=len(config.label_list))

        self.preprocessor: Preprocessor = PREPROCESSORS[config.wrapper_type](
            self,
            config.task_name,
            config.pattern_id)

        # WARP
        # init verbalizer
        if config.wrapper_type == "mlm" and config.init_verbalizer:
            label_id_2_idx = {}
            for label in self.preprocessor.label_map:
                label_id = self.preprocessor.label_map[label]
                idx = self.tokenizer.encode(self.preprocessor.pvp.verbalize(label), add_special_tokens=False)[0]

                label_id_2_idx[label_id] = idx

            self.model.init_verbalizer(label_id_2_idx)

        self.embeddings = self.get_embeddings()
        
        self.prompt_encoder = PromptEncoder(
            config,
            prompt_length=self.preprocessor.pvp.prompt_len,
            embeddings = self.embeddings)

        self.train_dataloader = None
        self.eval_dataloader = None
        self.device = torch.device(config.device)

        self.model.to(self.device)
        self.prompt_encoder.to(self.device)
        self.task_helper = TASK_HELPERS[self.config.task_name](self) if self.config.task_name in TASK_HELPERS else None


    def get_embeddings(self) -> nn.Embedding:
        return get_embeddings(self.base_model, self.config.model_type)

    def generate_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate inputs for pretrained model, replace part of the input embedding by Continuous Prompt ."""

        input_ids, prompt_indexes = batch['input_ids'], batch['prompt_indexes']

        inputs_embeds = self.prompt_encoder(input_ids, prompt_indexes)

        inputs = {
            'inputs_embeds': inputs_embeds,
            'attention_mask': batch["attention_mask"],
            'labels': batch["labels"]
        }

        if self.config.model_type == "bert":
            inputs["token_type_ids"] = batch["token_type_ids"]

        if self.config.wrapper_type in ["mlm", "rtd"]:
            inputs["label_flag"] = batch["label_flag"]

        return inputs

    def train(self,
              train_data: List[InputExample],
              train_batch_size: int,
              eval_data: List[InputExample],
              eval_batch_size: int,
              
              num_train_epochs: int = 3,
              learning_rate: float = 1e-5,
              weight_decay: float = 0.01,
              adam_epsilon: float = 1e-8,
              warmup_steps: int = 0,
              gradient_accumulation_steps: int = 1,
              max_grad_norm: float = 1.0,
              logging_steps: int = 50,
              eval_every_step: int = 100,
              base_model_learning_rate: float = None) -> None:

        if self.train_dataloader is None:
            self.train_dataloader = self.preprocessor.generate_train_dataloader(train_data, train_batch_size // gradient_accumulation_steps)

        train_dataloader = self.train_dataloader

        one_epoch_steps = len(train_dataloader) // gradient_accumulation_steps
        total_steps = one_epoch_steps * num_train_epochs

        model = self.model
        prompt_encoder = self.prompt_encoder

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay,
                'lr': base_model_learning_rate if base_model_learning_rate is not None else learning_rate
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': base_model_learning_rate if base_model_learning_rate is not None else learning_rate
            },

            {'params': prompt_encoder.prompt_ctx_embedding.parameters()},
            {'params': prompt_encoder.prompt_embeddings.parameters()},
            {'params': prompt_encoder.lstm_head.parameters()},
            {'params': prompt_encoder.mlp_head.parameters()}
        ]
        # embedding_parameters = prompt_encoder.parameters()

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_avg_metric = 0

        if self.config.zero_shot:
            logger.info("--- zero-shot result ---")
            logger.info(self.evaluate(eval_data, eval_batch_size))

        for epoch in trange(int(num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.train()
                prompt_encoder.train()

                batch = {k: t.to(self.device) for k, t in batch.items()}

                loss = None

                loss = self.task_helper.train_step(batch) if self.task_helper else None
                if loss is None:
                    loss = self.train_step(batch)

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % gradient_accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / logging_steps
                        logs['loss'] = loss_scalar
                        logging_loss = tr_loss
                        logger.info(json.dumps({**logs, **{'step': global_step}}))

                    if eval_every_step < one_epoch_steps and global_step % eval_every_step == 0:
                        scores = self.evaluate(eval_data, eval_batch_size)
                        avg_metric = compute_avg_metric(scores)
                        best_avg_metric = max(best_avg_metric, avg_metric)
                        scores["best_avg_metric"] = best_avg_metric
                        if self.config.save_model:
                            self.save(step=global_step)
                        logger.info(scores)

            if eval_every_step >= one_epoch_steps:
                # eval every epoch
                scores = self.evaluate(eval_data, eval_batch_size)
                avg_metric = compute_avg_metric(scores)
                best_avg_metric = max(best_avg_metric, avg_metric)
                scores["best_avg_metric"] = best_avg_metric
                if self.config.save_model:
                    self.save(step=global_step)
                logger.info(scores)

        # eval after training
        if eval_every_step < one_epoch_steps:
            scores = self.evaluate(eval_data, eval_batch_size)
            avg_metric = compute_avg_metric(scores)
            best_avg_metric = max(best_avg_metric, avg_metric)
            scores["best_avg_metric"] = best_avg_metric
            if self.config.save_model:
                self.save(step=global_step)
            logger.info(scores)

    def _eval(self, eval_data: List[InputExample], eval_batch_size: int) -> Dict:
        if self.eval_dataloader is None:
            self.eval_dataloader = self.preprocessor.generate_eval_dataloader(eval_data, eval_batch_size)

        eval_dataloader = self.eval_dataloader
        preds = None
        out_label_ids, question_ids = None, None
        eval_losses = []

        self.model.eval()
        self.prompt_encoder.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: t.to(self.device) for k, t in batch.items()}
            labels = batch['labels']
            with torch.no_grad():

                logits = self.task_helper.eval_step(batch) if self.task_helper else None
                if logits is None:
                    outputs = self.eval_step(batch)
                    logits = outputs.logits
                    eval_loss = outputs.loss

                eval_losses.append(eval_loss.item())

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
                if 'question_idx' in batch:
                    question_ids = batch['question_idx'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
                if 'question_idx' in batch:
                    question_ids = np.append(question_ids, batch['question_idx'].detach().cpu().numpy(), axis=0)

        return {
            "eval_loss": np.mean(eval_losses),
            'preds': preds,
            'labels': out_label_ids,
            'question_ids': question_ids
        }

    def evaluate(self, eval_data: List[InputExample], eval_batch_size: int) -> Dict:
        results = self._eval(eval_data, eval_batch_size)

        predictions = np.argmax(results['preds'], axis=1)
        scores = {}
        metrics = METRICS.get(self.config.task_name, DEFAULT_METRICS)
        for metric in metrics:
            if metric == 'acc':
                scores[metric] = simple_accuracy(predictions, results['labels'])
            elif metric == 'f1':
                scores[metric] = f1_score(results['labels'], predictions)
            elif metric == 'f1-macro':
                scores[metric] = f1_score(results['labels'], predictions, average='macro')
            elif metric == 'em':
                scores[metric] = exact_match(predictions, results['labels'], results['question_ids'])
            else:
                raise NotImplementedError(f"Metric '{metric}' not implemented")

        scores["eval_loss"] = results["eval_loss"]

        return scores

    def predict(self, test_data: List[InputExample], test_batch_size: int) -> np.ndarray:
        self.model.eval()
        results = self._eval(test_data, test_batch_size)

        predictions = np.argmax(results['preds'], axis=1)

        return predictions

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.generate_inputs(batch)
        outputs = self.model(**inputs)

        return outputs.loss

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        inputs = self.generate_inputs(batch)
        outputs = self.model(**inputs)

        return outputs

    def mlm_train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a MLM training step."""
        inputs = self.generate_inputs(batch)
        outputs = self.model(**inputs)
        return outputs.loss

    def save(self, step: int) -> None:
        path = self.config.output_dir
        task_name = self.config.task_name

        if not os.path.exists(path):
            os.mkdir(path)

        logger.info("Saving models.")

        state = {
            "prompt_encoder": self.prompt_encoder.state_dict(),
            "model": self.model.state_dict()
        }

        save_path_file = os.path.join(path, "%s_step%s.pth" % (task_name, step))
        torch.save(state, save_path_file)

    def load(self, save_path_file: str) -> None:
        state = torch.load(save_path_file)
        self.prompt_encoder.load_state_dict(state["prompt_encoder"])
        self.model.load_state_dict(state["model"])

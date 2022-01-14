from typing import Dict

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

class PromptDecoderForMLM(nn.Module):
    def __init__(self, mlm: PreTrainedModel, num_labels: int):
        super(PromptDecoderForMLM, self).__init__()

        self.mlm = mlm
        self.num_labels = num_labels
        self.loss_fn = nn.CrossEntropyLoss()

        hidden_size = self.mlm.config.embedding_size \
            if hasattr(self.mlm.config, "embedding_size") \
            else self.mlm.config.hidden_size

        # verbalizer
        self.cls = nn.Linear(hidden_size, num_labels, bias=True)
        self.cls.bias.data.zero_()

        # self.lm_head = self.mlm.cls.predictions

    # WARP
    def init_verbalizer(self, label_id_2_idx: Dict[int, int]) -> None:
        for label_id, idx in label_id_2_idx.items():
            with torch.no_grad():
                self.cls.weight[label_id] = self.mlm.get_output_embeddings().weight[idx]
                self.cls.bias[label_id] = self.mlm.get_output_embeddings().bias[idx]

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        label_flag: torch.Tensor = None,
        labels: torch.Tensor = None
    )  -> ModelOutput:


        hidden_states = self.mlm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).hidden_states

        cls_logits = self.cls(hidden_states[label_flag > 0])

        loss = self.loss_fn(cls_logits, labels)

        outputs = {
            'logits': cls_logits,
            'loss': loss
        }

        return ModelOutput(outputs)


class PromptDecoderForNSP(nn.Module):
    def __init__(self, nsp: PreTrainedModel, num_labels: int):
        super(PromptDecoderForNSP, self).__init__()

        self.nsp = nsp
        self.num_labels = num_labels
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        labels: torch.Tensor = None
    )  -> ModelOutput:

        # (batch_size * num_labels, seq_len)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # (batch_size * num_labels, seq_len, hidden_size)
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None

        logits = self.nsp(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).logits

        # (batch_size, num_labels)
        cls_logits = logits[:, 0].view(-1, self.num_labels)
        # - 0 indicates sequence B is a continuation of sequence A,

        loss = self.loss_fn(cls_logits, labels)
        outputs = {
            'logits': cls_logits,
            'loss': loss
        }

        return ModelOutput(outputs)

class PromptDecoderForRTD(nn.Module):
    def __init__(self, rtd: PreTrainedModel, num_labels: int):
        super(PromptDecoderForRTD, self).__init__()

        self.rtd = rtd
        self.num_labels = num_labels

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        label_flag: torch.Tensor = None,
        labels: torch.Tensor = None
    ) -> ModelOutput:

        # (batch_size * num_labels, seq_len)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # (batch_size * num_labels, seq_len, hidden_size)
        inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None


        seq_len = input_ids.size(-1) if input_ids is not None else inputs_embeds.size(-2)

        logits = self.rtd(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).logits


        # (batch_size, num_labels, seq_len)
        reshaped_logits = logits.view(-1, self.num_labels, seq_len)

        # (batch_size, num_labels)
        reshaped_logits = reshaped_logits[label_flag > 0].view(-1, self.num_labels)
        cls_logits = -reshaped_logits

        loss = self.loss_fn(cls_logits, labels)

        outputs = {
            'logits': cls_logits,
            'loss': loss
        }

        return ModelOutput(outputs)

# PromptDecoders
MODELS = {
    'mlm': PromptDecoderForMLM,
    'nsp': PromptDecoderForNSP,
    'rtd': PromptDecoderForRTD
}

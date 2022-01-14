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

from typing import List
from abc import ABC, abstractmethod

import torch
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler

from data_utils.utils import InputFeatures, InputExample
from data_utils.pvp import PVP, PVPS

from src.utils import DictDataset

import log

logger = log.get_logger('root')

class Preprocessor(ABC):
    """
    A preprocessor that transforms an :class:`InputExample` into a :class:`InputFeatures` object so that it can be
    processed by the model being used.
    """

    def __init__(self, wrapper, task_name: str, pattern_id: int = 0):
        """
        Create a new preprocessor.

        :param wrapper: the wrapper for the language model to use
        :param task_name: the name of the task
        :param pattern_id: the id of the PVP to be used
        """
        self.wrapper = wrapper
        self.pvp = PVPS[task_name](self.wrapper, pattern_id)  # type: PVP
        self.label_map = {label: i for i, label in enumerate(self.wrapper.config.label_list)}

        self.task_helper = None

    @abstractmethod
    def get_input_features(self, example: InputExample) -> InputFeatures:
        """Convert the given example into a set of input features"""
        pass

    def _convert_examples_to_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example {}".format(ex_index))
            input_features = self.get_input_features(example)

            if self.wrapper.task_helper:
                self.wrapper.task_helper.add_special_input_features(example, input_features)
            features.append(input_features)
        return features

    def _generate_dataset(self, examples: List[InputExample]) -> DictDataset:
        features = self._convert_examples_to_features(examples)

        feature_dict = {
            'input_ids': torch.tensor([f.input_ids for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f.attention_mask for f in features], dtype=torch.long),
            'token_type_ids': torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
            'labels': torch.tensor([f.label for f in features], dtype=torch.long),
            'prompt_indexes': torch.tensor([f.prompt_indexes for f in features], dtype=torch.long),
        }

        if self.wrapper.config.wrapper_type in ["mlm", "rtd"]:
            feature_dict["label_flag"] = torch.tensor([f.label_flag for f in features], dtype=torch.long)

        if self.wrapper.task_helper:
            self.wrapper.task_helper.add_features_to_dict(features, feature_dict)
        return DictDataset(**feature_dict)

    def generate_train_dataloader(self, train_data: List[InputExample], train_batch_size: int) -> DataLoader:
        train_dataset = self._generate_dataset(train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        return train_dataloader

    def generate_eval_dataloader(self, eval_data: List[InputExample], eval_batch_size: int) -> DataLoader:
        eval_dataset = self._generate_dataset(eval_data)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size)
        return eval_dataloader


class MLMPreprocessor(Preprocessor):
    """Preprocessor for models pretrained using a masked language modeling objective (e.g., BERT)."""

    def get_input_features(self, example: InputExample) -> InputFeatures:

        input_ids, token_type_ids, prompt_indexes, label_index = self.pvp.encode(example)

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        if padding_length < 0:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label] if example.label is not None else -100

        label_flag = [0] * len(input_ids)
        label_flag[label_index] = 1

        return InputFeatures(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             prompt_indexes=prompt_indexes,
                             label=label,
                             label_flag=label_flag)

class NSPPreprocessor(Preprocessor):
    """Preprocessor for models pretrained using a next sentence prediction objective (e.g., BERT)."""
    def get_input_features(self, example: InputExample) -> InputFeatures:
        choices_input_ids, choices_token_type_ids, choices_prompt_indexes, _ = self.pvp.encode(example) # unnecessary label index
        choices_attention_mask = []

        for input_ids, token_type_ids in zip(choices_input_ids, choices_token_type_ids):
            attention_mask = [1] * len(input_ids)
            padding_length = self.wrapper.config.max_seq_length - len(input_ids)

            if padding_length < 0:
                raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

            input_ids += [self.wrapper.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            token_type_ids += [0] * padding_length

            choices_attention_mask.append(attention_mask)

            assert len(input_ids) == self.wrapper.config.max_seq_length
            assert len(attention_mask) == self.wrapper.config.max_seq_length
            assert len(token_type_ids) == self.wrapper.config.max_seq_length

        label = self.label_map[example.label] if example.label is not None else -100

        return InputFeatures(input_ids=choices_input_ids,
                             attention_mask=choices_attention_mask,
                             token_type_ids=choices_token_type_ids,
                             prompt_indexes=choices_prompt_indexes,
                             label=label)

class RTDPreprocessor(Preprocessor):
    """Preprocessor for models pretrained using a replaced token detection objective (e.g., ELECTRA)."""

    def get_input_features(self, example: InputExample) -> InputFeatures:

        choices_input_ids, choices_token_type_ids, choices_prompt_indexes, label_index = self.pvp.encode(example)
        choices_attention_mask = []

        for input_ids, token_type_ids in zip(choices_input_ids, choices_token_type_ids):
            attention_mask = [1] * len(input_ids)
            padding_length = self.wrapper.config.max_seq_length - len(input_ids)

            if padding_length < 0:
                raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

            input_ids += [self.wrapper.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            token_type_ids += [0] * padding_length

            choices_attention_mask.append(attention_mask)

            assert len(input_ids) == self.wrapper.config.max_seq_length
            assert len(attention_mask) == self.wrapper.config.max_seq_length
            assert len(token_type_ids) == self.wrapper.config.max_seq_length


        label = self.label_map[example.label] if example.label is not None else -100

        single_label_flag = [0] * len(input_ids)
        single_label_flag[label_index] = 1

        label_flag = [single_label_flag for _ in range(len(choices_input_ids))]

        return InputFeatures(input_ids=choices_input_ids,
                             attention_mask=choices_attention_mask,
                             token_type_ids=choices_token_type_ids,
                             prompt_indexes=choices_prompt_indexes,
                             label=label,
                             label_flag=label_flag)

PREPROCESSORS = {
    'mlm': MLMPreprocessor,
    'nsp': NSPPreprocessor,
    'rtd': RTDPreprocessor
}

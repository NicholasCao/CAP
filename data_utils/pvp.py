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

import random
from abc import ABC, abstractmethod
from typing import Tuple, List, Union

import torch
from transformers import PreTrainedTokenizer

from data_utils.utils import InputExample

import log

logger = log.get_logger('root')

class ShortenableText(str):
    def __init__(self, *args):
        super(ShortenableText, self).__init__()
        self.shortenable = True
        self._enable_truncate_head = False

    def enable_truncate_head(self):
        self._enable_truncate_head = True

class ShortenableList(list):
    def __init__(self, *args):
        super(ShortenableList, self).__init__(*args)
        self.shortenable = True
        self._enable_truncate_head = False

    def enable_truncate_head(self):
        self._enable_truncate_head = True

    def truncate(self):
        if self._enable_truncate_head:
            self.pop(0)
        else:
            self.pop()

class Meta(object):
    def __init__(self, key):
        self.key = key

    def get_text(self, example: InputExample):
        return example.meta[self.key]

Pattern = Tuple[List[Union[str, ShortenableText]], List[Union[str, ShortenableText]]]

DataForPreprocessor = Tuple[List[int], List[int], List[int], int] or Tuple[List[List[int]], List[List[int]], List[List[int]], int]

# Prompt-Verbalizer Patterns 
class PVP(ABC):

    def __init__(self, wrapper = None, pattern_id: int = 0, seed: int = 42):
        """
        Create a new PVP.

        :param wrapper: the wrapper for the underlying language model
        :param pattern_id: the pattern id to use
        :param seed: a seed to be used for generating random numbers if necessary
        """
        self.wrapper = wrapper
        self.pattern_id = pattern_id
        self.rng = random.Random(seed)
        self.labels = self.wrapper.config.label_list

        self.token_map = {}
        self.prompt_len = 0 # ignore pseudo token
        self.prompt_id = None

        self.parts = self.get_parts()
        self._pre_encode()

    @property
    def label(self) -> str:
        """Return the label token"""
        return "[LABEL]"

    @property
    def text_a(self) -> str:
        """Return the text_a token"""
        return "[TEXT_A]"

    @property
    def text_b(self) -> str:
        """Return the text_b token"""
        return "[TEXT_B]"

    @property
    def prompt_token(self) -> str:
        """Return the prompt token, will encode with lstm"""
        return "[PROMPT_TOKEN]"

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.wrapper.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.wrapper.tokenizer.mask_token_id

    @staticmethod
    def meta(key: str) -> Meta:
        return Meta(key)

    @staticmethod
    def shortenable(text: str, enable_truncate_head: bool = False) -> ShortenableText:
        """Return an instance of this string that is marked as shortenable"""
        text = ShortenableText(text)
        if enable_truncate_head:
            text.enable_truncate_head()
        return text

    @staticmethod
    def is_shortenable(x: any) -> bool:
        """Check whether the text is shortenable or not"""
        return isinstance(x, ShortenableText) or isinstance(x, ShortenableList)

    def _is_prompt_token(self, x: any) -> bool:
        return x == self.prompt_token

    def _is_text(self, x: any) -> bool:
        return x == self.text_a or x == self.text_b

    def _is_label(self, x: any) -> bool:
        return x == self.label

    def _is_meta(self, x: any) -> bool:
        return isinstance(x, Meta)

    def _seq_length(self, parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x in parts if not only_shortenable or self.is_shortenable(x)]) if parts else 0

    def _truncate_last(self, parts: List[Union[List, ShortenableList]]):
        last_idx = max(idx for idx, seq in enumerate(parts) if self.is_shortenable(seq) and seq)
        parts[last_idx].truncate()

    def truncate(self, parts_a: List[Union[str, ShortenableText]], parts_b:List[Union[str, ShortenableText]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._truncate_last(parts_a)
            else:
                self._truncate_last(parts_b)

    @abstractmethod
    def get_parts(self) -> Pattern:
        """
        Return pattern for the task.
        Each part should have only one token, except the text part.

        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    def verbalize(self, label) -> str:
        return self.VERBALIZER[label]

    def _convert_to_pseudo_tokens_ids(self, part: str, tokenizer: PreTrainedTokenizer) -> List[int]:
        tokens = tokenizer.tokenize(part)

        for i, token in enumerate(tokens):
            if token != "▁": # for albert
                if token not in self.token_map:
                    new_pseudo_token =  "[V%d]" % len(self.token_map)
                    tokenizer.add_tokens(new_pseudo_token)
                    self.token_map[token] = new_pseudo_token
                tokens[i] = self.token_map[token]

        return tokenizer.convert_tokens_to_ids(tokens)

    def _assign_embeddings(self) -> None:
        """Init all pseudo tokens' embeddings with raw tokens' embedding"""

        logger.info("assigning embeddings")
        tokenizer = self.wrapper.tokenizer

        self.wrapper.base_model.resize_token_embeddings(len(tokenizer))

        # must get embeddings after resize_token_embeddings
        embeddings = self.wrapper.get_embeddings()

        for token, pseudo_token in self.token_map.items():
            id_a = tokenizer.encode(pseudo_token, add_special_tokens=False)[0]
            id_b = tokenizer.encode(token, add_special_tokens=False)[0]

            with torch.no_grad():
                embeddings.weight[id_a] = embeddings.weight[id_b].detach().clone()
            logger.info("assigned < %s = %s >" % (pseudo_token, token))

    def _pre_encode(self) -> None:
        """Prepare for encoding"""

        tokenizer: PreTrainedTokenizer = self.wrapper.tokenizer
        tokenizer.add_tokens(self.prompt_token)
        parts_a,  parts_b = self.parts

        self.prompt_id = tokenizer.encode(self.prompt_token, add_special_tokens=False)[0]
        # tokenizer.convert_tokens_to_ids([self.prompt_token])[0]

        for i, part in enumerate(parts_a):
            if self._is_text(part) or self._is_label(part) or self._is_meta(part):
                continue
            elif self._is_prompt_token(part):
                self.prompt_len += 1
                parts_a[i] = [self.prompt_id]
            else:
                parts_a[i] = self._convert_to_pseudo_tokens_ids(part, tokenizer)

        for i, part in enumerate(parts_b):
            if self._is_text(part) or self._is_label(part) or self._is_meta(part):
                continue
            elif self._is_prompt_token(part):
                self.prompt_len += 1
                parts_b[i] = [self.prompt_id]
            else:
                parts_b[i] = self._convert_to_pseudo_tokens_ids(part, tokenizer)

        for label in self.labels:
            self._convert_to_pseudo_tokens_ids(self.verbalize(label), tokenizer)

        self._assign_embeddings()

    def _encode_single_choice(self, example: InputExample, label_id: int) -> Tuple[List[int], List[int], List[int], int]:
        tokenizer: PreTrainedTokenizer = self.wrapper.tokenizer

        parts_a, parts_b = self.parts

        new_parts_a, new_parts_b = [], []

        text_a = example.text_a # .rstrip(string.punctuation)
        text_b = example.text_b # .rstrip(string.punctuation)

        for part in parts_a:
            if self._is_text(part):
                tokens = tokenizer.encode(
                    text_a if part == self.text_a 
                    else text_b,
                    add_special_tokens=False)

                if self.is_shortenable(part):
                    tokens = ShortenableList(tokens)
                    if part._enable_truncate_head:
                        tokens.enable_truncate_head()

                new_parts_a.append(tokens)
            elif self._is_label(part):
                new_parts_a.append([label_id])
            elif self._is_meta(part):
                new_parts_a.append(tokenizer.encode(part.get_text(example), add_special_tokens=False))
            else:
                new_parts_a.append(part)

        for part in parts_b:
            if self._is_text(part):
                tokens = tokenizer.encode(
                    text_a if part == self.text_a 
                    else text_b,
                    add_special_tokens=False)
                if self.is_shortenable(part):
                    tokens = ShortenableList(tokens)
                    if part._enable_truncate_head:
                        tokens.enable_truncate_head()

                new_parts_b.append(tokens)
            elif self._is_label(part):
                new_parts_b.append([label_id])
            elif self._is_meta(part):
                new_parts_b.append(tokenizer.encode(part.get_text(example), add_special_tokens=False))
            else:
                new_parts_b.append(part)
        
        parts_a, parts_b = new_parts_a, new_parts_b

        num_special = self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length - num_special)

        tokens_a = [token_id for part in parts_a for token_id in part]
        tokens_b = [token_id for part in parts_b for token_id in part] if parts_b else []

        if tokens_b:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
        else:
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a)

        prompt_indexes = [i for i in range(len(input_ids)) if input_ids[i] == self.prompt_id]
        assert len(prompt_indexes) == self.prompt_len

        return input_ids, token_type_ids, prompt_indexes, input_ids.index(label_id)

    def encode(self, example: InputExample) -> DataForPreprocessor:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :return: A tuple, consisting of a list of input ids, token type ids and label index (for mlm and rtd)
        """
        wrapper_type = self.wrapper.config.wrapper_type
        tokenizer: PreTrainedTokenizer = self.wrapper.tokenizer

        if wrapper_type == "mlm":
            return self._encode_single_choice(example, self.mask_id)

        if wrapper_type == "nsp" or wrapper_type == "rtd":
            choices_input_ids = []
            choices_token_type_ids = []
            choices_prompt_indexes = []

            for label in self.labels:
                label_id = self._convert_to_pseudo_tokens_ids(self.verbalize(label), tokenizer)[0]
                input_ids, token_type_ids, prompt_indexes, label_index = self._encode_single_choice(example, label_id)
                choices_input_ids.append(input_ids)
                choices_token_type_ids.append(token_type_ids)
                choices_prompt_indexes.append(prompt_indexes)

            return choices_input_ids, choices_token_type_ids, choices_prompt_indexes, label_index
        raise NotImplementedError(f"Wrapper Type '{wrapper_type}' not implemented")


class RtePVP(PVP):

    VERBALIZER = {
        "not_entailment": "No",
        "entailment": "Yes"
    }

    def get_parts(self) -> Pattern:
        text_a = self.shortenable(self.text_a, enable_truncate_head=True)
        text_b = self.text_b

        if self.pattern_id == 0:
            parts_a = [text_a, "?"]
            parts_b = [self.label, ",", text_b, "."]
            return parts_a, parts_b

        elif self.pattern_id == 1:
            parts_a = [text_a, "?"]
            parts_b = [self.label, self.prompt_token, text_b, "."]

            return parts_a, parts_b

        elif self.pattern_id == 2:
            parts_a = [self.prompt_token, text_a, self.prompt_token]
            parts_b = [self.prompt_token, self.label, self.prompt_token, text_b, self.prompt_token]

            return parts_a, parts_b
        else:
            raise ValueError("unknown pattern_id.")


class CbPVP(PVP):

    VERBALIZER = {
        "contradiction": "No",
        "entailment": "Yes",
        "neutral": "Maybe"
    }

    def get_parts(self) -> Pattern:
        text_a = self.shortenable(self.text_a, enable_truncate_head=True)
        text_b = self.text_b

        if self.pattern_id == 0:
            parts_a = [text_a, "?"]
            parts_b = [self.label, ",", text_b, "."]
            return parts_a, parts_b

        elif self.pattern_id == 1:
            parts_a = [text_a, "?"]
            parts_b = [self.label, self.prompt_token, text_b, "."]

            return parts_a, parts_b

        elif self.pattern_id == 2:
            parts_a = [self.prompt_token, text_a, self.prompt_token]
            parts_b = [self.prompt_token, self.label, self.prompt_token, text_b, self.prompt_token]

            return parts_a, parts_b
        else:
            raise ValueError("unknown pattern_id.")

class BoolQPVP(PVP):

    VERBALIZER = {
        "False": "No",
        "True": "Yes"
    }

    def get_parts(self) -> Pattern:
        text_a = self.shortenable(self.text_a, enable_truncate_head=True)
        text_b = self.text_b

        if self.pattern_id == 0:
            parts_a = [text_a, "."]
            parts_b = [text_b, "?", self.label, "."]
            return parts_a, parts_b

        elif self.pattern_id == 1:
            parts_a = [text_a, "."]
            parts_b = [text_b, "?", self.prompt_token, self.label, "."]

            return parts_a, parts_b

        elif self.pattern_id == 2:
            parts_a = [self.prompt_token, text_a, self.prompt_token]
            parts_b = [self.prompt_token, text_b, self.prompt_token, self.label, self.prompt_token]

            return parts_a, parts_b
        else:
            raise ValueError("unknown pattern_id.")

class WicPVP(PVP):

    VERBALIZER = {
        "False": "No",
        "True": "Yes"
    }

    def get_parts(self) -> Pattern:
        text_a = self.shortenable(self.text_a, enable_truncate_head=True)
        text_b = self.text_b
        word = self.meta("word")

        if self.pattern_id == 0:
            parts_a = [text_a, "."]
            parts_b = [text_b, ".", "the", word, "?", self.label, "."]
            return parts_a, parts_b

        elif self.pattern_id == 1:
            parts_a = [text_a, "."]
            parts_b = [text_b, ".", self.prompt_token, word, "?", self.label, "."]

            return parts_a, parts_b

        elif self.pattern_id == 2:
            parts_a = [self.prompt_token, text_a, self.prompt_token]
            parts_b = [self.prompt_token, text_b, self.prompt_token, self.prompt_token, word, self.prompt_token, self.label, self.prompt_token]
            return parts_a, parts_b

        else:
            raise ValueError("unknown pattern_id.")

class WscPVP(PVP):

    VERBALIZER = {
        "False": "No",
        "True": "Yes"
    }

    def get_parts(self) -> Pattern:
        text_a = self.shortenable(self.text_a, enable_truncate_head=True)

        pronoun = self.meta("span2_text")
        target = self.meta("span1_text")

        if self.pattern_id == 0:
            parts_a = [text_a, "."]
            parts_b = ["The pronoun", "\"", pronoun, "\"", "refers to",  "\"", target, "\"", "?", self.label, "."]
            return parts_a, parts_b

        elif self.pattern_id == 1:
            parts_a = [text_a, "."]
            parts_b = ["The pronoun", "\"", pronoun, "\"", "refers to",  "\"", target, "\"", self.prompt_token, self.label, "."]
            return parts_a, parts_b

        elif self.pattern_id == 2:
            parts_a = [text_a, "."]

            parts_b = [self.prompt_token, self.prompt_token, target, 
                self.prompt_token, self.prompt_token, pronoun, self.prompt_token, self.prompt_token, self.label, "."]

            return parts_a, parts_b
        else:
            raise ValueError("unknown pattern_id.")

class MultiRcPVP(PVP):

    VERBALIZER = {
        "0": "No",
        "1": "Yes"
    }

    def get_parts(self) -> Pattern:
        text_a = self.shortenable(self.text_a, enable_truncate_head=True)
        text_b = self.text_b
        answer = self.meta("answer")

        if self.pattern_id == 0:
            parts_a = [text_a, "."]
            parts_b = ["Question: ", text_b, "? Is it", answer, "?", self.label, "."]
            return parts_a, parts_b

        elif self.pattern_id == 1:
            parts_a = [text_a, "."]
            parts_b = [text_b, "? ", self.prompt_token, '"', answer, '"', self.prompt_token, "?", self.label, "."]
            return parts_a, parts_b

        elif self.pattern_id == 2:
            parts_a = [text_a, self.prompt_token]
            parts_b = [self.prompt_token, text_b, self.prompt_token, self.prompt_token, answer, self.prompt_token, self.label, self.prompt_token]
            return parts_a, parts_b
        else:
            raise ValueError("unknown pattern_id.")

PVPS = {
    'rte': RtePVP,
    'cb': CbPVP,
    'boolq': BoolQPVP,
    'wic': WicPVP,
    'wsc': WscPVP,
    "multirc": MultiRcPVP,
}

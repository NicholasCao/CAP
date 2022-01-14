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

"""
This file contains the logic for loading data for all tasks.
"""
import string
import csv
import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict

import log
from data_utils.utils import InputExample

logger = log.get_logger('root')

class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading train/dev/test examples for a given task.
    """
    def __init__(self, data_dir: str, log_example: bool = False):
        self.data_dir = data_dir
        self.log_example = log_example

    @abstractmethod
    def get_train_examples(self, num_example: int = -1) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass

    @staticmethod
    def _read_tsv(file, quotechar=None) -> List[str]:
        """Reads a tab separated value file."""
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines[1:] # ignore the head

    @staticmethod
    def _read_json(file) -> List[Dict[str, str]]:
        """Reads a json file."""
        with open(file, 'r', encoding='utf-8') as f:
            dicts = []
            for l in f:
                dicts.append(json.loads(l))
            return dicts

class GlueProcessor(DataProcessor):
    def get_train_examples(self, num_example: int = -1) -> List[InputExample]:
        num_example = None if num_example == -1 else num_example

        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "train.tsv"))[:num_example], "train")

    def get_dev_examples(self):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "dev.tsv")), "dev")

    def get_test_examples(self):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "test.tsv")), "test")

    def get_labels(self):
        raise NotImplementedError()

    def _create_examples(self, lines: List[str], set_type: str) -> List[InputExample]:
        raise NotImplementedError()

class SuperGlueProcessor(DataProcessor):
    def get_train_examples(self, num_example: int = -1) -> List[InputExample]:
        num_example = None if num_example == -1 else num_example

        return self._create_examples(
            self._read_json(os.path.join(self.data_dir, "train.jsonl"))[:num_example], "train")

    def get_dev_examples(self):
        return self._create_examples(
            self._read_json(os.path.join(self.data_dir, "val.jsonl")), "dev")

    def get_test_examples(self):
        return self._create_examples(
            self._read_json(os.path.join(self.data_dir, "test.jsonl")), "test")

    def get_labels(self):
        raise NotImplementedError()

    def _create_examples(self, dicts: List[Dict[str, str]], set_type: str) -> List[InputExample]:
        raise NotImplementedError()

class CbProcessor(SuperGlueProcessor):
    """Processor for the CB data set."""

    def get_labels(self):
        return ["entailment", "contradiction", "neutral"]

    def _create_examples(self, dicts: List[Dict[str, str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, example_json) in enumerate(dicts):
            guid = "%s-%s" % (set_type, i)
            text_a = example_json["premise"].rstrip(string.punctuation)
            text_b = example_json["hypothesis"].rstrip(string.punctuation)

            label = example_json.get("label")

            if self.log_example:
                if i == 0:
                    logger.info("*** Data Example ***")
                    logger.info("guid: %s" % guid)
                    logger.info("text_a: %s" % text_a)
                    logger.info("text_b: %s" % text_b)
                    logger.info("label: %s\n" % label)

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)
        return examples

class RteProcessor(CbProcessor):
    """Processor for the RTE data set (SuperGLUE version)."""

    def get_labels(self) -> List[str]:
        return ["not_entailment", "entailment"]

class BoolQProcessor(SuperGlueProcessor):
    """Processor for the BoolQ data set."""

    def get_labels(self):
        return ["False", "True"]

    def _create_examples(self, dicts: List[Dict[str, str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, example_json) in enumerate(dicts):
            guid = "%s-%s" % (set_type, i)
            text_a = example_json["passage"].rstrip(string.punctuation)
            text_b = example_json["question"].rstrip(string.punctuation)

            label = example_json.get("label")
            label = str(label) if label is not None else None

            if self.log_example:
                if i == 0:
                    logger.info("*** Data Example ***")
                    logger.info("guid: %s" % guid)
                    logger.info("text_a: %s" % text_a)
                    logger.info("text_b: %s" % text_b)
                    logger.info("label: %s\n" % label)

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)
        return examples

class WicProcessor(SuperGlueProcessor):
    """Processor for the WiC data set."""

    def get_labels(self):
        return ["False", "True"]

    def _create_examples(self, dicts: List[Dict[str, str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, example_json) in enumerate(dicts):
            guid = "%s-%s" % (set_type, i)

            text_a = example_json['sentence1'].rstrip(string.punctuation)
            text_b = example_json['sentence2'].rstrip(string.punctuation)
            meta = {'word': example_json['word']}
            label = example_json.get("label")
            label = str(label) if label is not None else None

            if self.log_example:
                if i == 0:
                    logger.info("*** Data Example ***")
                    logger.info("guid: %s" % guid)
                    logger.info("text_a: %s" % text_a)
                    logger.info("text_b: %s" % text_b)
                    logger.info("label: %s" % label)
                    logger.info("meta: %s\n" % meta)

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, meta=meta)
            examples.append(example)
        return examples

class WscProcessor(SuperGlueProcessor):
    """Processor for the WSC data set."""

    def get_labels(self):
        return ["False", "True"]

    def _create_examples(self, dicts: List[Dict[str, str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, example_json) in enumerate(dicts):
            guid = "%s-%s" % (set_type, i)

            text_a = example_json['text'].rstrip(string.punctuation)
            meta = {
                'span1_text': example_json['target']['span1_text'].rstrip(string.punctuation),
                'span2_text': example_json['target']['span2_text'].rstrip(string.punctuation),
            }
            label = example_json.get("label")
            label = str(label) if label is not None else None


            if self.log_example:
                if i == 0:
                    logger.info("*** Data Example ***")
                    logger.info("guid: %s" % guid)
                    logger.info("text_a: %s" % text_a)
                    logger.info("label: %s" % label)
                    logger.info("meta: %s\n" % meta)

            example = InputExample(guid=guid, text_a=text_a, label=label, meta=meta)
            examples.append(example)
        return examples

class MultiRcProcessor(SuperGlueProcessor):
    """Processor for the MultiRC data set."""

    def get_labels(self):
        return ["0", "1"]

    def _create_examples(self, dicts: List[Dict[str, str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, example_json) in enumerate(dicts):
            text = example_json['passage']['text'].rstrip(string.punctuation)
            questions = example_json['passage']['questions']
            
            for question_json in questions:
                question = question_json["question"].rstrip(string.punctuation)
                question_idx = question_json['idx']
                answers = question_json["answers"]
                for answer_json in answers:
                    label = answer_json.get("label")
                    label = str(label) if label is not None else None
                    answer_idx = answer_json["idx"]
                    answer = answer_json["text"].rstrip(string.punctuation)
                    guid = f'{set_type}-{i}-q{question_idx}-a{answer_idx}'
                    meta = {
                        'question_idx': question_idx,
                        'answer_idx': answer_idx,
                        'answer': answer
                    }
                    example = InputExample(guid=guid, text_a=text, text_b=question, label=label, meta=meta)
                    examples.append(example)

        if self.log_example:
            example = examples[0]
            logger.info("*** Data Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("text_a: %s" % example.text_a)
            logger.info("text_b: %s" % example.text_b)
            logger.info("label: %s" % example.label)
            logger.info("meta: %s\n" % example.meta)

        return examples
    # def _create_examples(path: str, set_type: str) -> List[InputExample]:
    #     examples = []

    #     with open(path, encoding='utf8') as f:
    #         for line in f:
    #             example_json = json.loads(line)

    #             passage_idx = example_json['idx']
    #             text = example_json['passage']['text']
    #             questions = example_json['passage']['questions']
    #             for question_json in questions:
    #                 question = question_json["question"]
    #                 question_idx = question_json['idx']
    #                 answers = question_json["answers"]
    #                 for answer_json in answers:
    #                     label = str(answer_json["label"]) if 'label' in answer_json else None
    #                     answer_idx = answer_json["idx"]
    #                     guid = f'{set_type}-p{passage_idx}-q{question_idx}-a{answer_idx}'
    #                     meta = {
    #                         'passage_idx': passage_idx,
    #                         'question_idx': question_idx,
    #                         'answer_idx': answer_idx,
    #                         'answer': answer_json["text"]
    #                     }
    #                     idx = [passage_idx, question_idx, answer_idx]
    #                     example = InputExample(guid=guid, text_a=text, text_b=question, label=label, meta=meta, idx=idx)
    #                     examples.append(example)

    #     question_indices = list(set(example.meta['question_idx'] for example in examples))
    #     label_distribution = Counter(example.label for example in examples)
    #     logger.info(f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
    #                 f"distribution {list(label_distribution.items())}")
    #     return examples

PROCESSORS = {
    "cb": CbProcessor,
    "rte": RteProcessor,
    "boolq": BoolQProcessor,
    "wic": WicProcessor,
    "wsc": WscProcessor,
    "multirc": MultiRcProcessor,
}


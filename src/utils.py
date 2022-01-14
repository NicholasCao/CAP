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

import json
import random
from collections import defaultdict
from typing import Dict
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel

def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def exact_match(predictions: np.ndarray, actuals: np.ndarray, question_ids: np.ndarray) -> float:
    """Compute the exact match (EM) for a sequence of predictions and actual labels"""
    unique_questions = set(question_ids)

    q_actuals = list(zip(question_ids, actuals))
    q_predictions = list(zip(question_ids, predictions))

    actuals_per_question = defaultdict(list)
    predictions_per_question = defaultdict(list)

    for qid, val in q_actuals:
        actuals_per_question[qid].append(val)
    for qid, val in q_predictions:
        predictions_per_question[qid].append(val)

    em = 0
    for qid in unique_questions:
        if actuals_per_question[qid] == predictions_per_question[qid]:
            em += 1
    em /= len(unique_questions)

    return em

def simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return (preds == labels).mean()

def compute_avg_metric(scores):
    total, cnt = 0.0, 0
    for k, v in scores.items():
        if k != "eval_loss":
            total += v
            cnt += 1
    return total / cnt if cnt > 0 else 0

class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)

def get_embeddings(model: PreTrainedModel, model_type: str) -> nn.Embedding:
    """Return the word embeddings of PrLms"""
    if model_type == "bert":
        return model.bert.embeddings.word_embeddings
    elif model_type == "roberta":
        return model.roberta.embeddings.word_embeddings
    elif model_type == "electra":
        return model.electra.embeddings.word_embeddings
    elif model_type == "deberta":
        return model.deberta.embeddings.word_embeddings
    elif model_type == "albert":
        return model.albert.embeddings.word_embeddings
    else:
        raise NotImplementedError(f"Model Type '{model_type}' not implemented")

class Config(OrderedDict):
    """Base class for args"""
    def __getitem__(self, k):
        inner_dict = {k: v for (k, v) in self.items()}
        return inner_dict[k]

    def __setitem__(self, key, value):
        super().__setattr__(key, value)


    def __repr__(self):
        return str(self.to_json_string())

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.__dict__, indent=2, sort_keys=True) + "\n"
    
    def read_config(self, path: str):
        with open(path) as f:
            config = json.load(f)
            super().__init__(config)
    
    def update(self, config: Dict) -> None:
        for k, v in config.items():
            self.__setitem__(k, v)

    def insert_default(self, default_config: Dict) -> None:
        for k, v in default_config.items():
            if hasattr(self, k):
                continue
            self.__setitem__(k, v)

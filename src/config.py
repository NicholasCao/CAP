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

class WrapperConfig(object):
    """A configuration for a :class:`TransformerModelWrapper`."""

    def __init__(self,
                 wrapper_type: str,
                 model_type: str,
                 model_name_or_path: str,
                 task_name: str,
                 max_seq_length: int,
                 label_list: List[str],
                 device: str = None,
                 pattern_id: int = 0,
                 output_dir: str = "output",
                 hidden_size: int = 128,
                 unique_prompt: bool = True,
                 eval_every_step: int = 100,
                 init_verbalizer: bool = True,
                 zero_shot: bool = False,
                 save_model: bool = False):

        self.wrapper_type = wrapper_type
        self.model_type = model_type
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.label_list = label_list
        self.device = device
        self.pattern_id = pattern_id
        self.output_dir = output_dir
        self.hidden_size = hidden_size
        self.eval_every_step = eval_every_step
        self.unique_prompt = unique_prompt

        self.init_verbalizer = init_verbalizer
        self.zero_shot = zero_shot
        self.save_model = save_model

    def __repr__(self):
        return repr(self.__dict__)

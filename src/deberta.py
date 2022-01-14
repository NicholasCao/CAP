from transformers import DebertaV2PreTrainedModel, DebertaV2Model

import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import ModelOutput

class DebertaMaskPredictions(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.classifer = nn.Linear(config.hidden_size, 1)

        self.config = config

    def forward(self, hidden_states):
        ctx_states = hidden_states[:, 0, :]
        hidden_states = self.LayerNorm(ctx_states.unsqueeze(-2) + hidden_states)

        # hidden_states = self.LayerNorm(hidden_states)

        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)

        logits = self.classifer(hidden_states).squeeze(-1)

        return logits

# replaced token detection(RTD)
class DebertaV3ForRTD(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaV2Model(config)
        self.mask_predictions = DebertaMaskPredictions(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        token_type_ids=None
    ):
        outputs = self.deberta(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        sequence_output = outputs[0]
        logits = self.mask_predictions(sequence_output)

        return ModelOutput({'logits': logits})

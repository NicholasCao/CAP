import torch
import torch.nn as nn

from src.config import WrapperConfig

class PromptEncoder(nn.Module):
    def __init__(self,
                 config: WrapperConfig,
                 prompt_length: int,
                 embeddings: nn.Embedding):
        super(PromptEncoder, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.prompt_length = prompt_length

        self.embeddings = embeddings
        self.prompt_ctx_embedding = nn.Embedding(embeddings.num_embeddings, embeddings.embedding_dim)
        self.prompt_ctx_embedding.weight.data.zero_() # init with zero

        # alation experiment (NES)
        # self.prompt_ctx_embedding.weight.data.copy_(self.embeddings.weight.data)

        self.embedding_size = embeddings.embedding_dim

        self.prompt_embeddings = nn.Embedding(self.prompt_length, self.embedding_size)

        self.lstm_head = nn.LSTM(input_size=self.embedding_size,
                                 hidden_size=self.hidden_size // 2,
                                 num_layers=2,
                                 bidirectional=True,
                                 batch_first=True)

        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.embedding_size))

        self.device = torch.device(config.device)

    def forward(
        self, 
        input_ids: torch.Tensor = None,
        prompt_indexes: torch.Tensor = None
    ) -> torch.Tensor:

        # (batch_size, prompt_length)
        # assert prompt_indexes.size(1) == self.prompt_length

        bz = input_ids.size(0)

        input_ids = input_ids.view(-1, input_ids.size(-1))

        raw_embeds = self.embeddings(input_ids)

        if self.prompt_length == 0:
            return raw_embeds
        
        prompt_indexes = prompt_indexes.view(-1, prompt_indexes.size(-1))

        prompt_ids = torch.LongTensor(list(range(self.prompt_length))).to(self.device)

        # (prompt_length, embedding_size)
        prompt_embeds = self.prompt_embeddings(prompt_ids)

        if self.config.unique_prompt:
            # Gradient-Disentangled Embedding Sharing
            # stop grad to raw_embeds, just update prompt_ctx_embedding
            all_embeddings = raw_embeds.detach().clone() + self.prompt_ctx_embedding(input_ids)

            # alation experiment (NES)
            # all_embeddings = self.prompt_ctx_embedding(input_ids)

            for bidx in range(bz):
                for i in range(self.prompt_length):
                    all_embeddings[bidx, prompt_indexes[bidx, i], :] = prompt_embeds[i, :]

            all_embeddings = self.lstm_head(all_embeddings)[0]
            all_embeddings = self.mlp_head(all_embeddings)

            # embeddings to replace the raw embeddings
            for bidx in range(bz):
                for i in range(self.prompt_length): 
                    raw_embeds[bidx, prompt_indexes[bidx, i], :] = all_embeddings[bidx, prompt_indexes[bidx, i], :] \
                        + prompt_embeds[i, :] # skip connection

            return raw_embeds

        else:
            prompt_embeds = self.lstm_head(prompt_embeds.unsqueeze(0))[0]
            prompt_embeds = self.mlp_head(prompt_embeds)

            prompt_embeds = prompt_embeds.squeeze(0)

            for bidx in range(bz):
                for i in range(self.prompt_length):
                    raw_embeds[bidx, prompt_indexes[bidx, i], :] = prompt_embeds[i, :]

            return raw_embeds

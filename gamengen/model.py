import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.modeling_utils import ModelMixin
from torch import nn


class ActionEmbeddingModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, action_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(action_ids)

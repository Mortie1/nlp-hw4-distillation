import torch
from torch import nn


class FactorizedEmbeddings(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, factor=128, pretrained_embeds=None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.factor = factor
        self.weight1 = nn.Embedding(num_embeddings, factor)
        self.weight2 = nn.Linear(factor, embedding_dim, bias=False)
        if pretrained_embeds is not None:
            U, S, Vh = torch.linalg.svd(
                pretrained_embeds[:, :embedding_dim], full_matrices=False
            )
            S = S[:factor]
            self.weight1.weight.data = U[:, :factor] @ torch.diag(S)
            self.weight2.weight.data = Vh[:factor].T

    def forward(self, x):
        return self.weight2(self.weight1(x))
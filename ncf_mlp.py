import torch
import torch.nn as nn
from typing import List

class NCF_MLP(nn.Module):
    """
    Multi-Layer Perceptron for Neural Collaborative Filtering.
    """
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64, hidden_layers: List[int] = [128, 64, 32]):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        layers = []
        input_dim = emb_dim * 2
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        user_vec = self.user_emb(user_idx)
        item_vec = self.item_emb(item_idx)
        features = torch.cat([user_vec, item_vec], dim=-1)
        out = self.mlp(features).squeeze() * 5.0
        return out
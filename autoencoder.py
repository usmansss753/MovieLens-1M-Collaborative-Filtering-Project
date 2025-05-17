import torch
import torch.nn as nn
from typing import List

class AutoencoderCF(nn.Module):
    """
    Autoencoder model for collaborative filtering.
    """
    def __init__(self, input_dim: int, latent_dim: int = 128, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        # Encoder
        enc_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev_dim, h))
            enc_layers.append(nn.ReLU())
            prev_dim = h
        enc_layers.append(nn.Linear(prev_dim, latent_dim))
        enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)
        # Decoder
        dec_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev_dim, h))
            dec_layers.append(nn.ReLU())
            prev_dim = h
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon
import torch
import torch.nn as nn
from torch.optim import Adam
from src.models.ncf_mlp import NCF_MLP
from src.models.autoencoder import AutoencoderCF
from torch.utils.data import DataLoader
from typing import Any

def fit_ncf(model: NCF_MLP, loader: DataLoader, epochs: int = 10, lr: float = 0.001, device: str = 'cpu') -> NCF_MLP:
    """
    Trains the NCF_MLP model using the provided DataLoader.
    """
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for u, m, r in loader:
            u, m, r = u.to(device), m.to(device), r.to(device)
            optimizer.zero_grad()
            preds = model(u, m)
            loss = loss_fn(preds, r)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[NCF] Epoch {ep+1}: Avg Loss = {running_loss / len(loader):.4f}")
    return model

def fit_autoencoder(model: AutoencoderCF, loader: DataLoader, n_users: int, n_movies: int, epochs: int = 10, lr: float = 0.001, device: str = 'cpu') -> AutoencoderCF:
    """
    Trains the AutoencoderCF model using the full user-item matrix.
    """
    model = model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    # Build user-item matrix
    user_item = torch.zeros(n_users, n_movies).to(device)
    for u, m, r in loader.dataset:
        user_item[u, m] = r
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon = model(user_item)
        loss = loss_fn(recon, user_item)
        loss.backward()
        optimizer.step()
        print(f"[Autoencoder] Epoch {ep+1}: Loss = {loss.item():.4f}")
    return model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from src.data_loader import prepare_movielens_loaders
from src.models.ncf_mlp import NCF_MLP
from src.models.autoencoder import AutoencoderCF
from src.train import fit_ncf, fit_autoencoder
from typing import Dict

def compute_mae(model: torch.nn.Module, loader: torch.utils.data.DataLoader, model_kind: str, n_users: int, n_movies: int, device: str = 'cpu') -> float:
    """
    Calculates the mean absolute error (MAE) for a given model and test loader.
    """
    model.eval()
    abs_error_sum = 0.0
    total_count = 0
    with torch.no_grad():
        if model_kind == 'ncf':
            for u, m, r in loader:
                u, m, r = u.to(device), m.to(device), r.to(device)
                preds = model(u, m)
                abs_error_sum += torch.abs(preds - r).sum().item()
                total_count += len(r)
        elif model_kind == 'autoencoder':
            user_item = torch.zeros(n_users, n_movies).to(device)
            for u, m, r in loader.dataset:
                user_item[u, m] = r
            recon = model(user_item)
            for u, m, r in loader:
                u, m, r = u.to(device), m.to(device), r.to(device)
                preds = recon[u, m]
                abs_error_sum += torch.abs(preds - r).sum().item()
                total_count += len(r)
    return abs_error_sum / total_count if total_count > 0 else float('nan')

def write_results(metrics: Dict[str, float], out_path: str = 'results/results.txt') -> None:
    """
    Writes the evaluation results to a file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('Model\tMAE\n')
        for name, mae in metrics.items():
            f.write(f'{name}\t{mae:.4f}\n')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader, n_users, n_movies = prepare_movielens_loaders()

    ncf = NCF_MLP(n_users, n_movies).to(device)
    ncf = fit_ncf(ncf, train_loader, device=device)
    ncf_mae = compute_mae(ncf, test_loader, 'ncf', n_users, n_movies, device=device)

    autoenc = AutoencoderCF(n_movies).to(device)
    autoenc = fit_autoencoder(autoenc, train_loader, n_users, n_movies, device=device)
    auto_mae = compute_mae(autoenc, test_loader, 'autoencoder', n_users, n_movies, device=device)

    results = {'NCF_MLP': ncf_mae, 'Autoencoder': auto_mae}
    write_results(results)
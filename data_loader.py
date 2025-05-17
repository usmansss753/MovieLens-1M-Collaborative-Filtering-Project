import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

class RatingsDataset(Dataset):
    """Custom dataset for user-movie ratings."""
    def __init__(self, df: pd.DataFrame, use_mapped: bool = False):
        if use_mapped:
            self.user_indices = torch.tensor(df['user_idx'].values, dtype=torch.long)
            self.movie_indices = torch.tensor(df['movie_idx'].values, dtype=torch.long)
        else:
            self.user_indices = torch.tensor(df['userId'].values, dtype=torch.long)
            self.movie_indices = torch.tensor(df['movieId'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int):
        return self.user_indices[idx], self.movie_indices[idx], self.ratings[idx]

def prepare_movielens_loaders(filepath: Optional[str] = None) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Loads and processes the MovieLens 1M ratings data, mapping user and movie IDs to contiguous indices.
    Returns train/test DataLoaders and the number of unique users and movies.
    """
    if filepath is None:
        filepath = os.path.join('data', 'movielens_1m', 'ratings.dat')
    columns = ['userId', 'movieId', 'rating', 'timestamp']
    ratings_df = pd.read_csv(filepath, sep='::', names=columns, engine='python')
    # Map IDs to contiguous indices
    user_id_to_idx = {uid: idx for idx, uid in enumerate(ratings_df['userId'].unique())}
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(ratings_df['movieId'].unique())}
    ratings_df['user_idx'] = ratings_df['userId'].map(user_id_to_idx)
    ratings_df['movie_idx'] = ratings_df['movieId'].map(movie_id_to_idx)
    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    train_set = RatingsDataset(train_df, use_mapped=True)
    test_set = RatingsDataset(test_df, use_mapped=True)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    return train_loader, test_loader, len(user_id_to_idx), len(movie_id_to_idx)
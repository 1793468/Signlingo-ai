"""
PyTorch Dataset over preprocessed landmark sequences (.npz files).

Owner: Mariam Ashraf Tobar

Each .npz file is expected to contain:
    X: array of variable-length landmark sequences, each (T, 126)
    y: array of integer sign labels, aligned with X
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.normalize import pad_sequence


class SignDataset(Dataset):
    def __init__(self, split_dir: str, augment: bool = False):
        self.X = []
        self.y = []
        self.augment = augment

        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".npz"):
                continue
            data = np.load(os.path.join(split_dir, fname), allow_pickle=True)
            for seq, label in zip(data["X"], data["y"]):
                self.X.append(pad_sequence(seq))
                self.y.append(label)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

"""
pose_dataset.py

Goal:
Load pose landmark CSV
and convert it into
PyTorch tensors.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class PoseDataset(Dataset):

    def __init__(self, csv_file):

        # Load CSV
        self.data = pd.read_csv(csv_file)

        # Remove frame_id column
        self.features = self.data.drop(
            columns=["frame_id"]
        )

    def __len__(self):

        return len(self.features)

    def __getitem__(self, idx):

        sample = self.features.iloc[idx].values

        sample = torch.tensor(
            sample,
            dtype=torch.float32
        )

        return sample
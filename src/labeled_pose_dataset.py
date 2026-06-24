"""
labeled_pose_dataset.py

Temporary dataset for learning
PyTorch classification.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class LabeledPoseDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

        self.features = self.data.drop(
            columns=["frame_id"]
        )

        # Temporary label
        # All samples belong to class 0

        self.labels = [0] * len(self.features)

    def __len__(self):

        return len(self.features)

    def __getitem__(self, idx):

        x = torch.tensor(
            self.features.iloc[idx].values,
            dtype=torch.float32
        )

        y = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return x, y
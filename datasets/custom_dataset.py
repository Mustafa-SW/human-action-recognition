import pandas as pd
import torch
from torch.utils.data import Dataset

class ActionDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        features = torch.tensor(
            self.data.iloc[idx, :-1].values,
            dtype=torch.float32
        )

        label = torch.tensor(
            self.data.iloc[idx, -1],
            dtype=torch.long
        )

        return features, label
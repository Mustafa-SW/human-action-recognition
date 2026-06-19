import pandas as pd
import torch
from torch.utils.data import Dataset

# Dataset is an abstract class in pytorch which allows it to load 
# and process samples one by one

class ActionDataset(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file) 
    
    # pandas.read_csv returns pandas.data_frame
    # A DataFrame is a two-dimensional, size-mutable, and potentially heterogeneous
    # tabular data structure with labeled axes (rows and columns)


    def __len__(self):
        return len(self.data)
    
    # the __getitem__ method allows you to treat the dataset object as a Python list
    # allowing you to access its rows by using square bracket indexing 
    # here particularly this method returns features and labels of the row 
    # whose index is provided as pytorch tensors

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
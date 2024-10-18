from os import path
import pandas as pd
import torch
from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self, dataset_type: str):
        """
        Load "train" or "test" data
        """
        self.data = torch.tensor(
            pd.read_csv(
                path.join(
                    path.dirname(path.abspath(__file__)), f"mnist_{dataset_type}.csv"
                )
            ).to_numpy()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.data[idx, 1:], self.data[idx, 0]

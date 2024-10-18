from os import path
import torch
from torch.utils.data import Dataset, DataLoader


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class Cifar100(Dataset):
    def __init__(self, dataset_type: str, flatten: bool = False):
        """
        Load "train" or "test" data
        """
        file_data = unpickle(
            path.join(path.dirname(path.abspath(__file__)), dataset_type)
        )
        self.data = torch.tensor(file_data[b"data"])
        self.lables = torch.tensor(file_data[b"fine_labels"])

        if not flatten:
            self.data = self.data.view(-1, 3, 32, 32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.data[idx], self.lables[idx]

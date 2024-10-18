from os import path
import torch
from torch.utils.data import Dataset
import torchvision

def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class Cifar100(Dataset):
    data_folder = "cifar-100-python"

    def __init__(self, dataset_type: str, size = None, flatten: bool = False):
        """
        Load "train" or "test" data
        """
        file_data = unpickle(
            path.join(path.dirname(path.abspath(__file__)), self.data_folder, dataset_type)
        )
        self.data = torch.tensor(file_data[b"data"])
        self.lables = torch.tensor(file_data[b"fine_labels"])

        if not flatten:
            self.data = self.data.view(-1, 3, 32, 32)

        if size is not None:
            self.data = torchvision.transforms.Resize(size)(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.data[idx], self.lables[idx]

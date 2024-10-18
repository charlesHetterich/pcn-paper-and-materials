from os import path
import torch
from torch.utils.data import Dataset
import torchvision

def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class Cifar10(Dataset):

    data_folder = "cifar-10-batches-py"
    train_fn = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    test_fn = ["test_batch"]

    def __init__(self, dataset_type: str, size = None, flatten: bool = False):
        """
        Load "train" or "test" data
        """
        files = (
            self.train_fn
            if dataset_type == "train"
            else self.test_fn
            if dataset_type == "test"
            else None
        )
        if files is None:
            raise ValueError("dataset_type must be either `train` or `test`")

        self.data = []
        self.lables = []
        for fn in files:
            file_data = unpickle(path.join(path.dirname(path.abspath(__file__)), self.data_folder, fn))
            self.data.append(torch.tensor(file_data[b"data"]))
            self.lables.append(torch.tensor(file_data[b"labels"]))

        self.data = torch.cat(self.data)
        self.lables = torch.cat(self.lables)

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

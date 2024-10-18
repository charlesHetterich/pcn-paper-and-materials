from torch.utils.data import Dataset, DataLoader

from .mnist.DataLoader import MNIST
from .cifar10.DataLoader import Cifar10
from .cifar100.DataLoader import Cifar100


def load_data(dataset: Dataset, num_workers=0, batch_size=128):
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

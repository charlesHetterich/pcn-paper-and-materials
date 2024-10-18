import argparse
import os
from time import time

import torch
import torch.utils.tensorboard as tb

import numpy as np

import util
import datasets

class ConvNet(torch.nn.Module):
    def __init__(self, conv_layers: list[int], linear_layers: list[int], linear_type: str, pcn_dims: int = None):
        super().__init__()

        assert len(conv_layers) >= 2, "Must be at least 2 convolutional layers"
        assert len(linear_layers) >= 1, "Must be at least 1 linear layer"
        assert linear_type in ["mlp", "pcn"], "linear_type must be either 'mlp' or 'pcn'"
        if linear_type == "pcn":
            assert pcn_dims is not None, "pcn_dims must be specified if linear_type is 'pcn'"
            
        L = [
            torch.nn.Conv2d(conv_layers[0], conv_layers[1], 7, padding=3, stride=2),
            torch.nn.ReLU(),
        ]
        c = conv_layers[1]
        for l in conv_layers[2:]:
            L.append(torch.nn.Conv2d(c, l, 3, padding=1, stride=2))
            L.append(torch.nn.ReLU())
            c = l
        self.net = torch.nn.Sequential(*L)
        self.classifier = util.MLP([c, *linear_layers]) if linear_type == "mlp" \
            else util.PCN([c, *linear_layers], pcn_dims)

    def forward(self, x: torch.Tensor):
        z = self.net(x)
        z = torch.amax(z, dim=[2, 3])
        return self.classifier(z)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, required=True, help="number of epochs to train for.")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
parser.add_argument("--num_workers", type=int, required=True, help="Number of workers for data loading")
parser.add_argument("--dataset", type=str, required=True, choices=['cifar10', 'cifar100'], help="Dataset to use.")
parser.add_argument("--out_dir", type=str, required=True, help="Directory to save models to.")
args = parser.parse_args()

log_dir = "log_dir"
epochs = args.epochs
batch_size = args.batch_size
num_workers = args.num_workers
dataset_name = args.dataset
out_dir = args.out_dir

in_features = 3
out_features = 10 if dataset_name == "cifar10" else 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
train, test = None, None
if dataset_name == "cifar10":
    train = datasets.load_data(datasets.Cifar10("train"), batch_size=batch_size, num_workers=num_workers)
    test = datasets.load_data(datasets.Cifar10("test"), batch_size=batch_size, num_workers=num_workers)
else:
    train = datasets.load_data(datasets.Cifar100("train"), batch_size=batch_size, num_workers=num_workers)
    test = datasets.load_data(datasets.Cifar100("test"), batch_size=batch_size, num_workers=num_workers)

loss = torch.nn.CrossEntropyLoss()
conv_shape = [in_features, 32, 128, 512, 1024]
linear_shape = [1024, out_features]
models: list[tuple[str, torch.nn.Module]] = [
    (
        "ConvNet_mlp", 
        ConvNet(
            conv_layers=conv_shape,
            linear_layers=linear_shape,
            linear_type="mlp"
        ).to(device)),
    (
        "ConvNet_pcn16",
        ConvNet(
            conv_layers=conv_shape,
            linear_layers=linear_shape,
            linear_type="pcn",
            pcn_dims=16
        ).to(device),
    ),
]
model_utils = [
    (
        tb.SummaryWriter(log_dir=f"{log_dir}/{models[0][0]}"),
        [
            torch.optim.SGD(models[0][1].parameters(), lr=0.0001)
        ],
    ),
    (
        tb.SummaryWriter(log_dir=f"{log_dir}/{models[1][0]}"),
        [
            torch.optim.SGD(models[1][1].net.parameters(), lr=0.0001),
            util.pcnSGD(models[1][1].classifier, lr=0.0001, opp="log"),
        ],
    ),
]


# Train Loop
for epoch in range(epochs):
    for (_, model), (tb_logger, optimizers) in zip(models, model_utils):
        t = time()

        # train
        model.train()
        agg_acc = []
        for i, (x, y) in enumerate(train):
            step = epoch * len(train) + i
            x, y = x.to(device).float(), y.to(device)
            pred = model(x)

            l = loss(pred, y)
            l.backward()

            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

            agg_acc.append((pred.cpu().argmax(1) == y.cpu()).float().mean())
            tb_logger.add_scalar("loss", l, step)
        tb_logger.add_scalar("train_accuracy", np.mean(agg_acc), epoch)

        # validation
        model.eval()
        agg_acc = []
        with torch.no_grad():
            for i, (x, y) in enumerate(test):
                x, y = x.to(device).float(), y.to(device)
                pred = model(x)
                agg_acc.append((pred.cpu().argmax(1) == y.cpu()).float().mean())
            tb_logger.add_scalar("test_accuracy", np.mean(agg_acc), epoch)
        
        tb_logger.add_scalar("epoch_time", time() - t, epoch)

# save all models
os.makedirs(f"{out_dir}/{dataset_name}", exist_ok=True)
for name, model in models:
    torch.save(model.state_dict(), f"{out_dir}/{dataset_name}/{name}.pt")
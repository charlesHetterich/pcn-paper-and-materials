import argparse
import os
from time import time

import torch
import torch.utils.tensorboard as tb

import numpy as np

import util
import datasets


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

in_features = 16 * 16 * 3
out_features = 10 if dataset_name == "cifar10" else 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
train, test = None, None
if dataset_name == "cifar10":
    train = datasets.load_data(datasets.Cifar10("train", size=16), batch_size=batch_size, num_workers=num_workers)
    test = datasets.load_data(datasets.Cifar10("test", size=16), batch_size=batch_size, num_workers=num_workers)
else:
    train = datasets.load_data(datasets.Cifar100("train", size=16), batch_size=batch_size, num_workers=num_workers)
    test = datasets.load_data(datasets.Cifar100("test", size=16), batch_size=batch_size, num_workers=num_workers)

# Model definitions
loss = torch.nn.CrossEntropyLoss()
model_shape = [in_features, 1024, 1024, 2048, out_features]
models: list[tuple[str, torch.nn.Module]] = [
    (
        "LinearNet_mlp", 
        util.MLP(
            layers=model_shape,
        ).to(device)
    ),
    (
        "LinearNet_pcn4",
        util.PCN(
            layers=model_shape,
            dimensions=4
        ).to(device),
    ),
    (
        "LinearNet_pcn8",
        util.PCN(
            layers=model_shape,
            dimensions=8
        ).to(device),
    ),
    (
        "LinearNet_pcn16",
        util.PCN(
            layers=model_shape,
            dimensions=16
        ).to(device),
    ),
    (
        "LinearNet_pcn32",
        util.PCN(
            layers=model_shape,
            dimensions=32
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
            util.pcnSGD(models[1][1], lr=0.0001, opp="log"),
        ],
    ),
    (
        tb.SummaryWriter(log_dir=f"{log_dir}/{models[2][0]}"),
        [
            util.pcnSGD(models[2][1], lr=0.0001, opp="log"),
        ],
    ),
    (
        tb.SummaryWriter(log_dir=f"{log_dir}/{models[3][0]}"),
        [
            util.pcnSGD(models[3][1], lr=0.0001, opp="log"),
        ],
    ),
    (
        tb.SummaryWriter(log_dir=f"{log_dir}/{models[4][0]}"),
        [
            util.pcnSGD(models[4][1], lr=0.0001, opp="log"),
        ],
    ),
]

# Train Loop
for epoch in range(epochs):
    for (_, model), (tb_logger, optimizers) in zip(models, model_utils):
        t = time()

        # train
        model.train()
        agg_batch_acc = []
        for i, (x, y) in enumerate(train):
            step = epoch * len(train) + i
            x = x.view(-1, in_features)
            x, y = x.to(device).float(), y.to(device)
            pred = model(x)

            l = loss(pred, y)
            l.backward()

            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

            agg_batch_acc.append((pred.cpu().argmax(1) == y.cpu()).float().mean())
            tb_logger.add_scalar("loss", l, step)
        tb_logger.add_scalar("train_accuracy", np.mean(agg_batch_acc), epoch)

        # validation
        model.eval()
        agg_batch_acc = []
        with torch.no_grad():
            for i, (x, y) in enumerate(test):
                x = x.view(-1, in_features)
                x, y = x.to(device).float(), y.to(device)
                pred = model(x)
                agg_batch_acc.append((pred.cpu().argmax(1) == y.cpu()).float().mean())
            tb_logger.add_scalar("test_accuracy", np.mean(agg_batch_acc), epoch)
        
        tb_logger.add_scalar("epoch_time", time() - t, epoch)

# save all models
os.makedirs(f"{out_dir}/{dataset_name}", exist_ok=True)
for name, model in models:
    torch.save(model.state_dict(), f"{out_dir}/{dataset_name}/{name}.pt")
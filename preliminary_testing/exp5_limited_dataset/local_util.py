import torch
from torch import nn
import numpy as np

def tri(period: float, amplitude: float):
    """
    triangle wave function centered around 0 with period and amplitude
    """

    def triangle_wave_transform(x: torch.Tensor):
        # using sigal
        return (amplitude / period) * (
            (period - abs(x % (2 * period) - (1 * period)) - period / 2)
        )

    return triangle_wave_transform


# Model Definitions

class PCN(nn.Module):
    def __init__(
        self,
        layers: list[int],
        dimensions: int = 20,
    ):
        super().__init__()
        self.weight_transform = tri(0.1, 1)
        if len(layers) < 2:
            raise ValueError("At least 2 layers are required")

        self.layers = nn.ParameterList(
            [nn.Parameter(torch.rand(l, dimensions) * 2 - 1) for l in layers]
        )
        self.layers_bias = nn.ParameterList(
            [nn.Parameter((torch.rand(l, 1) * 2 - 1) * 0.1) for l in layers]
        )

    def forward(self, x: torch.Tensor):
        z = x
        for i, (l, lnext) in enumerate(zip(self.layers, self.layers[1:])):
            z = (
                z @ (self.weight_transform(torch.cdist(l, lnext)) / np.sqrt(l.shape[0]))
            ) + self.layers_bias[i + 1].T
            if i < len(self.layers) - 2:
                z = torch.relu(z)
        return z


# control model
class FCN(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        hidden: list[int],
    ):
        super().__init__()

        c = n_in
        L = []
        for l in hidden:
            L.append(nn.Linear(c, l))
            L.append(nn.ReLU())
            c = l

        L.append(nn.Linear(l, n_out))
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)
    
class pcnSGD(torch.optim.Optimizer):
    def __init__(self, model: nn.Module, lr=1e-3, opp=None):
        self.opp = opp
        self.model = model
        defaults = dict(lr=lr)
        super(pcnSGD, self).__init__(model.parameters(), defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # for param in 

        for group in self.param_groups:
            positions = group["params"][:int(len(group["params"])/2)]
            biases = group["params"][int(len(group["params"])/2):]

            for pos in positions:
                if pos.grad is None:
                    continue
                grad = pos.grad.data
                # apply custom gradient descent update
                
                if self.opp == None:
                    pos.data.add_(-group["lr"] * (pos.shape[0]), grad)
                elif self.opp == "log":
                    pos.data.add_(-group["lr"] * (pos.shape[0] / np.log2(pos.shape[0])), grad)
                elif self.opp == "sqrt":
                    pos.data.add_(-group["lr"] * (np.sqrt(pos.shape[0])), grad)


            for bias in biases:
                if bias.grad is None:
                    continue
                grad = bias.grad.data
                # apply custom gradient descent update
                bias.data.add_(-group["lr"] * 1e5, grad)

        return loss
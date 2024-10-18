import torch
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

class pcnSGD(torch.optim.Optimizer):
    def __init__(self, model: torch.nn.Module, lr=1e-3, opp=None):
        self.opp = opp
        defaults = dict(lr=lr)
        super().__init__(model.parameters(), defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            positions = group["params"][:int(len(group["params"])/2)]
            biases = group["params"][int(len(group["params"])/2):]

            for pos in positions:
                if pos.grad is None:
                    continue
                grad = pos.grad.data
                # apply custom gradient descent update
                
                if self.opp == None:
                    pos.data.add_(grad, alpha=-group["lr"] * (pos.shape[0]))
                elif self.opp == "log":
                    pos.data.add_(grad, alpha=-group["lr"] * (pos.shape[0] / np.log2(pos.shape[0])))
                elif self.opp == "sqrt":
                    pos.data.add_(grad, alpha=-group["lr"] * (np.sqrt(pos.shape[0])))


            for bias in biases:
                if bias.grad is None:
                    continue
                grad = bias.grad.data
                # apply custom gradient descent update
                bias.data.add_(grad, alpha=-group["lr"] * 1e5)

        return loss

class PCN(torch.nn.Module):
    def __init__(self, layers: list[int], dimensions: int = 20):
        super().__init__()

        assert len(layers) >= 2, "Must be at least 2 PCN layers"

        self.layer_params = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.rand(l, dimensions) * 2 - 1) for l in layers]
        )
        self.layer_biases = torch.nn.ParameterList(
            [torch.nn.Parameter((torch.rand(l, 1) * 2 - 1) * 0.1) for l in layers]
        )

    def forward(self, x: torch.Tensor):
        z = x
        for i, (l, lnext) in enumerate(zip(self.layer_params, self.layer_params[1:])):
            z = (
                z @ (tri(0.1, 1)(torch.cdist(l, lnext)) / np.sqrt(l.shape[0]))
            ) + self.layer_biases[i + 1].T
            if i < len(self.layer_params) - 2:
                z = torch.relu(z)
        return z


class ConvNet(torch.nn.Module):
    def __init__(self, conv_layers: list[int], linear_layers: list[int]):
        super().__init__()

        assert len(conv_layers) >= 2, "Must be at least 2 convolutional layers"
        assert len(linear_layers) >= 1, "Must be at least 1 linear layer"

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

        L = []
        for l in linear_layers[:-1]:
            L.append(torch.nn.Linear(c, l))
            L.append(torch.nn.ReLU())
            c = l
        L.append(torch.nn.Linear(c, linear_layers[-1]))
        L.append(torch.nn.Softmax(dim=1))
        self.classifier = torch.nn.Sequential(*L)

    def forward(self, x: torch.Tensor):
        z = self.net(x)
        z = torch.amax(z, dim=[2, 3])
        return self.classifier(z)

class ConvNetPCN(torch.nn.Module):
    def __init__(self, conv_layers: list[int], linear_layers: list[int], pcn_dims: int):
        super().__init__()

        assert len(conv_layers) >= 2, "Must be at least 2 convolutional layers"
        assert len(linear_layers) >= 1, "Must be at least 1 linear layer"

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
        self.classifier = PCN([c, *linear_layers], pcn_dims)
    
    def forward(self, x: torch.Tensor):
        z = self.net(x)
        z = torch.amax(z, dim=[2, 3])
        return torch.softmax(self.classifier(z), dim=1)
import torch
import numpy as np
# alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)

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

class PCN(torch.nn.Module): # this PCN implements dropout to match the original AlexNet
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
            # dropout
            if i < len(self.layer_params) - 2:
                z = torch.dropout(z, 0.5, self.training)

            # linear
            z = (
                z @ (tri(0.1, 1)(torch.cdist(l, lnext)) / np.sqrt(l.shape[0]))
            ) + self.layer_biases[i + 1].T
            
            # ReLU
            if i < len(self.layer_params) - 2:
                z = torch.relu(z)
        return z


class AlexNetPCN(torch.nn.Module):
    def __init__(self, pcn_dims: int):
        super().__init__()

        self.alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=False)
        self.alexnet.classifier = PCN([9216, 4096, 4096, 10], pcn_dims)
    
    def forward(self, x: torch.Tensor):
        return self.alexnet(x)
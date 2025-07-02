import torch
import torch.nn as nn
import numpy as np


class ICNN(nn.Module):
    def __init__(self, in_channels=3, dim_hidden=256, beta_softplus=100, alpha=0.0, pos_weights=False, rectifier_fn=torch.nn.ReLU(), device="cpu",ks=16):
        super().__init__()
        self.hidden = dim_hidden
        self.lin = nn.ModuleList([
            nn.Conv2d(in_channels, dim_hidden, 3, bias=True, stride=1, padding=1),
            nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=2, padding=1),
            nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=1, padding=1),
            nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=2, padding=1),
            nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=1, padding=1),
            nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=2, padding=1),
            nn.Conv2d(dim_hidden, 64, kernel_size=ks, bias=False, stride=1, padding=0),
            nn.Linear(64, 1),
        ])
        self.res = nn.ModuleList([
            nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1),
            nn.Conv2d(in_channels, dim_hidden, 3, stride=1, padding=1),
            nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1),
            nn.Conv2d(in_channels, dim_hidden, 3, stride=1, padding=1),
            nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1),
            nn.Conv2d(in_channels, 64, kernel_size=ks, stride=1, padding=0),
        ])
        self.act = nn.Softplus(beta=beta_softplus)
        self.alpha = alpha
        self.pos_weights = pos_weights
        if pos_weights:
            self.rectifier_fn = rectifier_fn
        if device is not None:
            self.to(device)

    def forward(self, x):
        bsize = x.shape[0]
        image_size = np.array([x.shape[-2], x.shape[-1]])
        y = x.clone()
        y = self.act(self.lin[0](y))
        size = [image_size, image_size // 2, image_size // 2, image_size // 4, image_size // 4, image_size // 8]
        if self.pos_weights:
            for core in self.lin[1:]:
                core.weight.data = self.rectifier_fn(core.weight.data)
        for core, res, (s_x, s_y) in zip(self.lin[1:-2], self.res[:-1], size[:-1]):
            x_scaled = nn.functional.interpolate(x, (s_x, s_y), mode="bilinear")
            y = self.act(core(y) + res(x_scaled))
        x_scaled = nn.functional.interpolate(x, tuple(size[-1]), mode="bilinear")
        y = self.lin[-2](y) + self.res[-1](x_scaled)
        y = self.act(y)
        y = torch.mean(y, dim=(2, 3))
        y = y.reshape(bsize, 64)
        y = self.lin[-1](y)
        y = y + self.alpha * x.reshape(x.shape[0], -1).pow(2).sum(1, keepdim=True)
        return y

    def init_weights(self, mean, std):
        print("init weights")
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.normal_(mean, std).exp_()

    def wclip(self):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    def grad(self, x):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            y = self.forward(x)
            grad = torch.autograd.grad(y.sum(), x, retain_graph=True, create_graph=True)[0]
        return grad
    

class ICNN_deep(nn.Module):
    r"""
    Input Convex Neural Network (ICNN) Profonda.

    :param int in_channels: Number of input channels.
    :param int dim_hidden: Number of hidden units.
    :param float beta_softplus: Beta parameter for the softplus activation function.
    :param float alpha: Strongly convex parameter.
    :param bool pos_weights: Whether to force positive weights in the forward pass.
    :param torch.nn.Module rectifier_fn: Activation function to use to force postive weight.
    :param str device: Device to use for the model.
    """

    def __init__(
        self,
        in_channels=3,
        dim_hidden=256,
        beta_softplus=100,
        alpha=0.0,
        pos_weights=False,
        rectifier_fn=torch.nn.Softplus(),
        device="cpu",
    ):
        super().__init__()

        self.hidden = dim_hidden

        self.lin = nn.ModuleList(
            [
                nn.Conv2d(in_channels, dim_hidden, 3, bias=True, stride=1, padding=1),  # 128
                nn.Conv2d(dim_hidden, dim_hidden, 3, bias=True, stride=2, padding=1),  # 64
                nn.Conv2d(dim_hidden, dim_hidden, 3, bias=True, stride=1, padding=1),  # 64
                nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=2, padding=1),  # 32
                nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=1, padding=1),  # 32
                nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=2, padding=1),  # 16
                nn.Conv2d(dim_hidden, dim_hidden, 3, bias=True, stride=1, padding=1),  # 16
                nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=2, padding=1),  # 8
                nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=1, padding=1),  # 8
                nn.Conv2d(dim_hidden, dim_hidden, 3, bias=True, stride=2, padding=1),  # 4
                nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=1, padding=1),  # 4
                nn.Conv2d(dim_hidden, 64, kernel_size=4, bias=False, stride=1, padding=0),  # 1
                nn.Linear(64, 1),
            ]
        )

        self.res = nn.ModuleList(
            [
                nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1),  # 64
                nn.Conv2d(in_channels, dim_hidden, 3, stride=1, padding=1),  # 64
                nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1),  # 32
                nn.Conv2d(in_channels, dim_hidden, 3, stride=1, padding=1),  # 32
                nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1),  # 16
                nn.Conv2d(in_channels, dim_hidden, 3, stride=1, padding=1),  # 16
                nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1),  # 8
                nn.Conv2d(in_channels, dim_hidden, 3, stride=1, padding=1),  # 8
                nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1),  # 4
                nn.Conv2d(in_channels, 64, kernel_size=4, stride=1, padding=0),  # 1
            ]
        )

        self.act = nn.Softplus(beta=beta_softplus)
        self.alpha = alpha
        self.pos_weights = pos_weights
        if pos_weights:
            self.rectifier_fn = rectifier_fn

        if device is not None:
            self.to(device)

    def forward(self, x):
        bsize = x.shape[0]
        image_size = np.array([x.shape[-2], x.shape[-1]])
        y = x.clone()
        y = self.act(self.lin[0](y))
        size = [
            image_size,
            image_size // 2,
            image_size // 2,
            image_size // 4,
            image_size // 4,
            image_size // 8,
            image_size // 8,
            image_size // 16,
            image_size // 16,
            image_size // 32,
            image_size // 32,
        ]

        if self.pos_weights:
            for core in self.lin[1:]:
                core.weight.data = self.rectifier_fn(core.weight.data)

        for core, res, (s_x, s_y) in zip(self.lin[1:-2], self.res[:-1], size[:-1]):
            x_scaled = nn.functional.interpolate(x, (s_x, s_y), mode="bilinear", align_corners=False)
            y = self.act(core(y) + res(x_scaled))

        x_scaled = nn.functional.interpolate(x, tuple(size[-1]), mode="bilinear", align_corners=False)
        y = self.lin[-2](y) + self.res[-1](x_scaled)
        y = self.act(y)
        

        y = torch.mean(y, dim=(2, 3))

        y = y.reshape(bsize, 64)
        y = self.lin[-1](y)  # (batch, 1)

        y = y + self.alpha * x.reshape(x.shape[0], -1).pow(2).sum(1, keepdim=True)

        return y

    def init_weights(self, mean, std):
        print("Initializing weights")
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.normal_(mean, std).exp_()

    def wclip(self):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    def grad(self, x):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            y = self.forward(x)
            grad = torch.autograd.grad(
                y.sum(), x, retain_graph=True, create_graph=True
            )[0]
        return grad

class ICNNWithBN(nn.Module):
    def __init__(self, in_channels=3, dim_hidden=256, beta_softplus=100, alpha=0.0, pos_weights=False, rectifier_fn=torch.nn.ReLU(), device="cpu"):
        super().__init__()
        self.hidden = dim_hidden
        self.lin = nn.ModuleList([
            nn.Conv2d(in_channels, dim_hidden, 3, bias=False, stride=1, padding=1),
            nn.BatchNorm2d(dim_hidden),
            nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(dim_hidden),
            nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=1, padding=1),
            nn.BatchNorm2d(dim_hidden),
            nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=2, padding=1),
            nn.BatchNorm2d(dim_hidden),
            nn.Conv2d(dim_hidden, dim_hidden, 3, bias=False, stride=1, padding=1),
            nn.BatchNorm2d(dim_hidden),
            nn.Conv2d(dim_hidden, 64, kernel_size=8, bias=False, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.Linear(64, 1, bias=False),
            nn.BatchNorm1d(1)
        ])
        self.res = nn.ModuleList([
            nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim_hidden),
            nn.Conv2d(in_channels, dim_hidden, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_hidden),
            nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim_hidden),
            nn.Conv2d(in_channels, dim_hidden, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_hidden),
            nn.Conv2d(in_channels, dim_hidden, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim_hidden),
            nn.Conv2d(in_channels, 64, kernel_size=8, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64)
        ])
        self.act = nn.Softplus(beta=beta_softplus)
        self.alpha = alpha
        self.pos_weights = pos_weights
        if pos_weights:
            self.rectifier_fn = rectifier_fn
        if device is not None:
            self.to(device)

    def forward(self, x):
        bsize = x.shape[0]
        image_size = np.array([x.shape[-2], x.shape[-1]])
        y = x.clone()
        y = self.act(self.lin[0](y))

        lin_layers = [self.lin[i:i+2] for i in range(1, len(self.lin) - 2, 2)]
        res_layers = [self.res[i:i+2] for i in range(0, len(self.res) - 2, 2)]
        target_sizes = [image_size // 2, image_size // 2, image_size // 4, image_size // 4, image_size // 8]

        if self.pos_weights:
            for i, core in enumerate(self.lin):
                if i > 0 and isinstance(core, nn.Conv2d):
                    core.weight.data = self.rectifier_fn(core.weight.data)

        for (core, bn), (res_conv, res_bn), size in zip(lin_layers, res_layers, target_sizes):
            x_scaled = nn.functional.interpolate(x, tuple(size), mode="bilinear")
            y = self.act(bn(core(y)) + res_bn(res_conv(x_scaled)))

        # Last convolutional block
        y = self.lin[-3](y) # The last Conv2d before the final linear
        y = self.lin[-2](y) # The BatchNorm after the last Conv2d

        x_scaled = nn.functional.interpolate(x, tuple(image_size // 8), mode="bilinear") # Match the last spatial size
        y = y + self.res[-2](x_scaled)
        y = self.res[-1](y)
        y = self.act(y)

        y = torch.mean(y, dim=(2, 3))
        y = y.reshape(bsize, 64)
        y = self.act(self.lin[-1](y))

        y = y + self.alpha * x.reshape(x.shape[0], -1).pow(2).sum(1, keepdim=True)
        return y


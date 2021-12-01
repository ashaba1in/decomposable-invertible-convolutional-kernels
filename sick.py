import math

import torch
from torch import nn
from torch.nn import functional as F


class SimpleDICK(nn.Module):

    def __init__(self, kernel_size: int = 3, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SimpleDICK, self).__init__()
        self.kernel_size = kernel_size
        self.horizontal_kernel = nn.Parameter(torch.empty(kernel_size, **factory_kwargs))
        self.vertical_kernel = nn.Parameter(torch.empty(kernel_size, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init_bounds = 1 / math.sqrt(self.kernel_size)
        nn.init.uniform_(self.horizontal_kernel, -init_bounds, init_bounds)
        nn.init.uniform_(self.vertical_kernel, -init_bounds, init_bounds)

    def _assemble_W(self):
        device = next(self.parameters()).device

        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim)).to(device)
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        x = F.conv2d(
            x,
            self.horizontal_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            padding=(0, self.kernel_size // 2)
        )
        x = F.conv2d(
            x, self.vertical_kernel.unsqueeze(-1).unsqueeze(0).unsqueeze(0),
            padding=(self.kernel_size // 2, 0)
        )
        log_det = 1
        return x, log_det

    def backward(self, z):
        return z

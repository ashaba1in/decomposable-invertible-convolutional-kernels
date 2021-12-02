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

    @staticmethod
    def recursive_log_det(kernel: torch.Tensor, size: int, blocks: int):
        mx = kernel.abs().max()
        b, a, c = kernel / mx
        bc = b * c
        det0 = 1
        det1 = a
        for i in range(2, size):
            det0, det1 = det1, a * det1 - bc * det0

        log_det = blocks * (size * torch.log(mx) + torch.log(torch.abs(det1) + 1e-8))
        # if not torch.isfinite(log_det):
        #     print(b, a, c, bc, mx, torch.log(mx), torch.log(torch.abs(det1) + 1e-8))
        #     det0 = 1
        #     det1 = a
        #     for i in range(2, size):
        #         det0, det1 = det1, a * det1 - bc * det0
        #         print(det1)
        return log_det

    def forward(self, x):
        _, _, h, w = x.shape
        x = F.conv2d(
            x,
            self.horizontal_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            padding=(0, self.kernel_size // 2)
        )
        x = F.conv2d(
            x, self.vertical_kernel.unsqueeze(-1).unsqueeze(0).unsqueeze(0),
            padding=(self.kernel_size // 2, 0)
        )

        # b, a, c = self.horizontal_kernel
        # D = a**2 - 4 * b * c
        # np1 = x.shape[-2] * x.shape[-1] + 1
        # log_apd = torch.log(a + torch.sqrt(D))
        # log_amd = torch.log(a - torch.sqrt(D))
        # log_det = -0.5 * torch.log(D) + (n + 1) * math.log(2) + \
        #           (n + 1) * log_apd + torch.log1p(-torch.exp((n + 1)))

        log_det = SimpleDICK.recursive_log_det(self.vertical_kernel, h, w) + \
                  SimpleDICK.recursive_log_det(self.horizontal_kernel, w, h)

        return x, log_det

    def backward(self, x):

        return x

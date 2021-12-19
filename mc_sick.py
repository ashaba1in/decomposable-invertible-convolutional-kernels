import torch
from torch import nn
from sick import SimpleDICK


class MultichannelDICK(nn.Module):
    def __init__(self, num_channels: int = 3, kernel_size: int = 3, device=None, dtype=None):
        super(MultichannelDICK, self).__init__()
        self.num_channels = num_channels
        self.kernels = nn.ModuleList([SimpleDICK(kernel_size, device, dtype) \
                                      for _ in range(num_channels)])

    def forward(self, x: torch.Tensor):
        y = []
        log_det = 0.0
        for i in range(self.num_channels):
            result, log_det_add = self.kernels[i].forward(x[:, i:i + 1])
            y += [result]
            log_det += log_det_add

        y = torch.cat(y, dim=1)
        return y, log_det

    def backward(self, y: torch.Tensor):
        x = []
        for i in range(self.num_channels):
            result = self.kernels[i].backward(y[:, i:i + 1])
            x += [result]

        x = torch.cat(x, dim=1)
        return x

import math

import torch
from torch import nn
from torch.nn import functional as F


def safe_log(x):
    return torch.log(torch.clamp(x, 1e-5))


class SimpleDICK(nn.Module):
    def __init__(self, kernel_size: int = 3, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SimpleDICK, self).__init__()
        assert kernel_size == 3
        self.kernel_size = kernel_size

        def init_kernel():
            init_bounds = 1 / math.sqrt(kernel_size)
            kernel = torch.empty(kernel_size, **factory_kwargs)
            kernel.uniform_(-init_bounds, init_bounds)

            diag_pos = self.kernel_size // 2
            kernel[diag_pos] = 4 * kernel[0].abs() * kernel[-1].abs() + 0.1 * torch.rand(1)
            return kernel

        self.horizontal_kernel = nn.Parameter(init_kernel())
        self.vertical_kernel = nn.Parameter(init_kernel())

    @torch.no_grad()
    def reset_parameters(self):
        init_bounds = 1 / math.sqrt(self.kernel_size)
        nn.init.uniform_(self.horizontal_kernel, -init_bounds, init_bounds)
        nn.init.uniform_(self.vertical_kernel, -init_bounds, init_bounds)

    @staticmethod
    def recursive_log_det(kernel: torch.Tensor, size: int, blocks: int):
        mx = kernel.abs().max() + 1e-7
        b, a, c = kernel / mx
        bc = b * c
        det0 = 1
        det1 = a
        for i in range(2, size + 1):
            det0, det1 = det1, a * det1 - bc * det0

        log_det = size * safe_log(mx) + safe_log(torch.abs(det1))
        # if not torch.isfinite(log_det):
        #     print(b, a, c, bc, mx, torch.log(mx), torch.log(torch.abs(det1) + 1e-8))
        #     det0 = 1
        #     det1 = a
        #     for i in range(2, size):
        #         det0, det1 = det1, a * det1 - bc * det0
        #         print(det1)
        return blocks * log_det

    @staticmethod
    def exact_log_det(kernel: torch.Tensor, size: int, blocks: int):
        b, a, c = kernel
        D = a ** 2 - 4 * b * c
        np1 = size + 1
        if D > 0:
            # print('pos')
            sqrtD = torch.sqrt(D)
            log_apd = safe_log(torch.abs(a + sqrtD))
            log_amd = safe_log(torch.abs(a - sqrtD))

            if a < 0:
                # print('a < 0')
                mx = log_amd
                delta = log_apd - log_amd
            else:
                # print('a > 0')
                mx = log_apd
                delta = log_amd - log_apd

            if -sqrtD < a < sqrtD and np1 % 2 == 1:
                # print('sign +')
                # remainder = F.softplus(np1 * delta, beta=1)
                remainder = torch.log1p(torch.exp(np1 * delta))
            else:
                # print('sign -')
                remainder = torch.log1p(-torch.exp(np1 * delta))
            # print(D, mx, delta, remainder)
            log_det = -0.5 * safe_log(D) - np1 * math.log(2) + \
                      np1 * mx + remainder

        elif D < 0:
            # print('neg')
            D = torch.abs(D)
            phi = torch.atan2(torch.sqrt(D), a)
            sin = torch.abs(torch.sin(np1 * phi))
            # print(D, a ** 2 + D, phi, sin)
            log_det = -0.5 * safe_log(D) - size * math.log(2) + np1 * 0.5 * safe_log(a ** 2 + D) + safe_log(sin)

        else:
            # print('zero')
            log_det = math.log(np1) - size * math.log(2) + size * safe_log(torch.abs(a))

        return log_det * blocks

    @staticmethod
    def constant_tridiagonal_algorithm(coeffs: torch.Tensor, d: torch.Tensor):
        a, b, c = coeffs
        n = d.shape[-1]
        b_prime = b.repeat(n)
        for i in range(1, n):
            w = a / b_prime[i - 1]
            b_prime[i] -= w * c
            d[..., i] = d[..., i] - w * d[..., i - 1]
        d[..., -1] /= b_prime[-1]
        for i in range(n - 2, -1, -1):
            d[..., i] = (d[..., i] - c * d[..., i + 1]) / b_prime[i]
        return d

    def forward(self, x):
        _, _, h, w = x.shape
        x = F.conv2d(
            x, self.horizontal_kernel.unsqueeze(0).unsqueeze(0).unsqueeze(0),
            padding=(0, self.kernel_size // 2)
        )
        x = F.conv2d(
            x, self.vertical_kernel.unsqueeze(-1).unsqueeze(0).unsqueeze(0),
            padding=(self.kernel_size // 2, 0)
        )

        # log_det = SimpleDICK.recursive_log_det(self.vertical_kernel, h, w) + \
        #           SimpleDICK.recursive_log_det(self.horizontal_kernel, w, h)

        log_det = SimpleDICK.exact_log_det(self.vertical_kernel, h, w) + \
                  SimpleDICK.exact_log_det(self.horizontal_kernel, w, h)
        return x, log_det

    def backward(self, x):
        # vertical inversion
        x = torch.transpose(x, -2, -1)
        x = SimpleDICK.constant_tridiagonal_algorithm(self.vertical_kernel, x)

        # transpose image back
        x = torch.transpose(x, -2, -1)

        # horizontal inversion
        x = SimpleDICK.constant_tridiagonal_algorithm(self.horizontal_kernel, x)
        return x

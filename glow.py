import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sick import SimpleDICK
from mc_sick import MultichannelDICK

class ActNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, 1, 1, requires_grad=True))
        self.t = nn.Parameter(torch.randn(1, dim, 1, 1, requires_grad=True))
        self.data_dep_init_done = False

    def forward(self, x):
        _, _, h, w = x.shape

        if not self.data_dep_init_done:
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)

            self.s.data = (-torch.log(flatten.std(dim=1))).view(1, -1, 1, 1).detach()
            self.t.data = -flatten.mean(dim=1).view(1, -1, 1, 1).detach()
            self.data_dep_init_done = True

        z = x * torch.exp(self.s) + self.t
        log_det = h * w * torch.sum(self.s)

        return z, log_det

    def backward(self, z):
        x = (z - self.t) * torch.exp(-self.s)
        return x


class Invertible1x1Conv(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        P, L, U = torch.lu_unpack(*Q.lu())
        self.P = nn.Parameter(P, requires_grad=False) # remains fixed during optimization
        self.L = nn.Parameter(L)  # lower triangular portion
        self.S = nn.Parameter(U.diag())  # "crop out" the diagonal to its own parameter
        self.U = nn.Parameter(torch.triu(U, diagonal=1))  # "crop out" diagonal, stored in S

    def _assemble_W(self):
        device = next(self.parameters()).device

        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim)).to(device)
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x):
        _, _, h, w = x.shape

        W = self._assemble_W()
        z = F.conv2d(x, W.unsqueeze(2).unsqueeze(3))
        log_det = h * w * torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def backward(self, z):
        W = self._assemble_W()
        W_inv = torch.inverse(W)
        x = F.conv2d(z, W_inv.unsqueeze(2).unsqueeze(3))
        return x


class ZeroConv2d(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_dim, 1, 1))

    def forward(self, x):
        z = F.pad(x, [1, 1, 1, 1], value=1)
        z = self.conv(z)
        z = z * torch.exp(self.scale * 3)

        return z


class Conv(nn.Module):
    def __init__(self, nin, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nin // 2, nh, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(nh, nh, 1),
            nn.ReLU(),
            ZeroConv2d(nh, nin)
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()

        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, x):
        return self.net(x)


class AffineCoupling(nn.Module):
    def __init__(self, dim, nh=512):
        super().__init__()

        self.net = Conv(dim, nh)

    def forward(self, x):
        x0, x1 = x.chunk(2, 1)

        log_s, t = self.net(x0).chunk(2, 1)
        s = torch.sigmoid(log_s + 2)
        z1 = (x1 + t) * s

        logdet = torch.sum(torch.log(s).view(x.shape[0], -1), 1)

        return torch.cat([x0, z1], 1), logdet

    def backward(self, z):
        z0, z1 = z.chunk(2, 1)

        log_s, t = self.net(z0).chunk(2, 1)
        s = torch.sigmoid(log_s + 2)
        x1 = z1 / s - t

        return torch.cat([z0, x1], 1)


class Flow(nn.Module):
    def __init__(self, dim, conv_type=0):
        super().__init__()

        self.actnorm = ActNorm(dim)
        if conv_type == 0:
            self.invconv = Invertible1x1Conv(dim)
        else:
            self.invconv = MultichannelDICK(dim)
        self.coupling = AffineCoupling(dim)

    def forward(self, x):
        z, logdet = self.actnorm(x)
        z, det1 = self.invconv(z)
        # z, det2 = self.dick(x)
        z, det3 = self.coupling(z)

        logdet = logdet + det1 + det3

        return z, logdet

    def backward(self, z):
        x = self.coupling.backward(z)
        # x = self.dick.backward(z)
        x = self.invconv.backward(x)
        x = self.actnorm.backward(x)

        return x


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * np.log(2 * np.pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, dim, n_flow, split=True):
        super().__init__()

        squeeze_dim = dim * 4 # remove // 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim, i % 2))

        self.split = split

        if split:
            self.prior = ZeroConv2d(dim * 2, dim * 4)
            # self.prior = ZeroConv2d(dim, dim)
        else:
            self.prior = ZeroConv2d(dim * 4, dim * 8)
            # self.prior = ZeroConv2d(dim, dim)

    def forward(self, x):
        b_size, n_c, h, w = x.shape
        squeezed = x.view(b_size, n_c, h // 2, 2, w // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        z = squeezed.contiguous().view(b_size, n_c * 4, h // 2, w // 2)
        # z = x
        logdet = 0

        for flow in self.flows:
            z, det = flow(z)
            logdet += det

        if self.split:
            z, z_new = z.chunk(2, 1)
            mean, log_sd = self.prior(z).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(z)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(z, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = z

        return z, logdet, log_p, z_new

    def backward(self, z, eps=None):
        x = z

        if self.split:
            mean, log_sd = self.prior(x).chunk(2, 1)
            z_new = gaussian_sample(eps, mean, log_sd)
            x = torch.cat([z, z_new], 1)

        else:
            zero = torch.zeros_like(x)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_sd)
            x = z

        for flow in self.flows[::-1]:
            x = flow.backward(x)

        b_size, n_c, h, w = x.shape

        unsqueezed = x
        unsqueezed = x.view(b_size, n_c // 4, 2, 2, h, w)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_c // 4, h * 2, w * 2
        )

        return unsqueezed


class Glow(nn.Module):
    def __init__(self, dim, n_flow, n_block):
        super().__init__()

        self.n_channels = dim
        self.n_block = n_block

        self.blocks = nn.ModuleList()
        n_channel = dim
        for i in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow))
            n_channel *= 2

        self.blocks.append(Block(n_channel, n_flow, split=False))

    def forward(self, x):
        log_p_sum = 0
        logdet = 0
        z = x
        z_outs = []

        for block in self.blocks:
            z, det, log_p, z_new = block(z)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return z_outs, log_p_sum, logdet

    def backward(self, z_list):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.backward(z_list[-1], z_list[-1])

            else:
                x = block.backward(x, z_list[-(i + 1)])

        return x

    def calc_z_shapes(self, input_size=32):
        n_channel = self.n_channels
        z_shapes = []

        for i in range(self.n_block - 1):
            input_size //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))

        return z_shapes

    def sample_z(self, size):
        device = next(self.parameters()).device

        z_shapes = self.calc_z_shapes()

        z_sample = []
        for z in z_shapes:
            z_new = torch.randn(size, *z) * 0.7
            z_sample.append(z_new.to(device))

        return z_sample

    @torch.no_grad()
    def sample(self, size):
        return self.backward(self.sample_z(size))


class SimpleFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.actnorm = ActNorm(dim)
        self.invconv = Invertible1x1Conv(dim)
        self.dick = MultichannelDICK(dim)
        # self.coupling = AffineCoupling(dim)

    def forward(self, x):
        z, logdet = self.actnorm(x)
        z, det1 = self.invconv(z)
        z, det2 = self.dick(x)
        
        # z, det2 = self.coupling(z)

        logdet = logdet + det1 + det2

        return z, logdet

    def backward(self, z):
        # x = self.coupling.backward(z)
        x = self.dick.backward(z)
        x = self.invconv.backward(x)
        x = self.actnorm.backward(x)

        return x

class SimpleGlow(nn.Module):
    def __init__(self, dim, n_flow, n_block):
        super().__init__()

        self.n_channels = dim
        self.n_block = n_block

        self.blocks = nn.ModuleList()
        n_channel = dim
        for i in range(n_block):
            self.blocks.append(SimpleFlow(n_channel))
            n_channel *= 2

    def forward(self, x):
        log_p_sum = 0
        logdet = 0
        z = x

        for block in self.blocks:
            z, det = block(z)
            logdet = logdet + det


        return z, logdet

    def backward(self, z):
        for i, block in enumerate(self.blocks[::-1]):
            z = block.backward(z)

        return z

    def calc_z_shapes(self, input_size=32):
        n_channel = self.n_channels
        z_shapes = []

        for i in range(self.n_block - 1):
            input_size //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))

        return z_shapes

    def sample_z(self, size):
        device = next(self.parameters()).device

        z_shapes = self.calc_z_shapes()

        z_sample = []
        for z in z_shapes:
            z_new = torch.randn(size, *z) * 0.7
            z_sample.append(z_new.to(device))

        return z_sample

    @torch.no_grad()
    def sample(self, size):
        return self.backward(self.sample_z(size))




class VanillaFlow(nn.Module):
    def __init__(self, dim, conv_type=0):
        super().__init__()

        self.actnorm = ActNorm(dim)
        if True:
            self.invconv = Invertible1x1Conv(dim)
        else:
            self.invconv = MultichannelDICK(dim)
        self.coupling = AffineCoupling(dim)

    def forward(self, x):
        z, logdet = self.actnorm(x)
        z, det1 = self.invconv(z)
        # z, det2 = self.dick(x)
        z, det3 = self.coupling(z)

        logdet = logdet + det1 + det3

        return z, logdet

    def backward(self, z):
        x = self.coupling.backward(z)
        # x = self.dick.backward(z)
        x = self.invconv.backward(x)
        x = self.actnorm.backward(x)

        return x


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * np.log(2 * np.pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class VanillaBlock(nn.Module):
    def __init__(self, dim, n_flow, split=True):
        super().__init__()

        squeeze_dim = dim * 4 # remove // 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(VanillaFlow(squeeze_dim, i % 2))

        self.split = split

        if split:
            self.prior = ZeroConv2d(dim * 2, dim * 4)
            # self.prior = ZeroConv2d(dim, dim)
        else:
            self.prior = ZeroConv2d(dim * 4, dim * 8)
            # self.prior = ZeroConv2d(dim, dim)

    def forward(self, x):
        b_size, n_c, h, w = x.shape
        squeezed = x.view(b_size, n_c, h // 2, 2, w // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        z = squeezed.contiguous().view(b_size, n_c * 4, h // 2, w // 2)
        # z = x
        logdet = 0

        for flow in self.flows:
            z, det = flow(z)
            logdet += det

        if self.split:
            z, z_new = z.chunk(2, 1)
            mean, log_sd = self.prior(z).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(z)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            log_p = gaussian_log_p(z, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = z

        return z, logdet, log_p, z_new

    def backward(self, z, eps=None):
        x = z

        if self.split:
            mean, log_sd = self.prior(x).chunk(2, 1)
            z_new = gaussian_sample(eps, mean, log_sd)
            x = torch.cat([z, z_new], 1)

        else:
            zero = torch.zeros_like(x)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_sd)
            x = z

        for flow in self.flows[::-1]:
            x = flow.backward(x)

        b_size, n_c, h, w = x.shape

        unsqueezed = x
        unsqueezed = x.view(b_size, n_c // 4, 2, 2, h, w)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_c // 4, h * 2, w * 2
        )

        return unsqueezed


class VanillaGlow(nn.Module):
    def __init__(self, dim, n_flow, n_block):
        super().__init__()

        self.n_channels = dim
        self.n_block = n_block

        self.blocks = nn.ModuleList()
        n_channel = dim
        for i in range(n_block - 1):
            self.blocks.append(VanillaBlock(n_channel, n_flow))
            n_channel *= 2

        self.blocks.append(VanillaBlock(n_channel, n_flow, split=False))

    def forward(self, x):
        log_p_sum = 0
        logdet = 0
        z = x
        z_outs = []

        for block in self.blocks:
            z, det, log_p, z_new = block(z)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return z_outs, log_p_sum, logdet

    def backward(self, z_list):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                x = block.backward(z_list[-1], z_list[-1])

            else:
                x = block.backward(x, z_list[-(i + 1)])

        return x

    def calc_z_shapes(self, input_size=32):
        n_channel = self.n_channels
        z_shapes = []

        for i in range(self.n_block - 1):
            input_size //= 2
            n_channel *= 2

            z_shapes.append((n_channel, input_size, input_size))

        input_size //= 2
        z_shapes.append((n_channel * 4, input_size, input_size))

        return z_shapes

    def sample_z(self, size):
        device = next(self.parameters()).device

        z_shapes = self.calc_z_shapes()

        z_sample = []
        for z in z_shapes:
            z_new = torch.randn(size, *z) * 0.7
            z_sample.append(z_new.to(device))

        return z_sample

    @torch.no_grad()
    def sample(self, size):
        return self.backward(self.sample_z(size))
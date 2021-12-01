import typing as tp

import torch
from torch import nn
from torchmetrics import Metric
from torchvision.models import inception_v3
from torchvision.transforms import Resize


def cov(
        x: torch.Tensor,
        x_mean: torch.Tensor = None
) -> torch.Tensor:
    dim = x.shape[-1]
    if x_mean is None:
        x_mean = x.mean(-1)
    x = x - x_mean
    return x @ x.T / (dim - 1)


@torch.no_grad()
def calculate_activation_statistics(
        activations: torch.Tensor,
) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    real_activations_mean = torch.mean(activations, dim=0)
    real_activations_cov = cov(activations, real_activations_mean)
    return real_activations_mean, real_activations_cov


def calculate_frechet_distance(
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: torch.Tensor,
        eps: float = 1e-8
) -> torch.Tensor:
    diff = mu1 - mu2
    regularizer = eps * torch.eye(*sigma1.shape, device=sigma1.device)
    eigenvals, _ = torch.eig((sigma1 + regularizer) @ (sigma2 + regularizer), eigenvectors=False)
    tr_covmean = torch.sum(torch.sqrt(torch.abs(eigenvals[:, 0])))

    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


class FidScore(Metric):
    _classifier = None

    def __init__(self, compute_on_step=False, dist_sync_on_step=False, classifier: tp.Optional[nn.Module] = None):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.classifier = classifier
        if self.classifier is None:
            if FidScore._classifier is None:
                import types
                self.resize = Resize((299, 299))

                @torch.no_grad()
                def get_activations(self_, x):
                    x = self.resize(x)
                    x = self_._forward(x)[0]
                    return x

                FidScore._classifier = inception_v3(pretrained=True, progress=False)
                FidScore._classifier.requires_grad_(False)
                FidScore._classifier.get_activations = types.MethodType(get_activations, FidScore._classifier)
            self.classifier = FidScore._classifier

        self.add_state("real_activations", default=[], dist_reduce_fx=None)
        self.add_state("fake_activations", default=[], dist_reduce_fx=None)

    @torch.no_grad()
    def update(self, real_images, fake_images) -> None:
        self.real_activations.append(self.classifier.get_activations(real_images))
        self.fake_activations.append(self.classifier.get_activations(fake_images))

    def compute(self):
        m1, s1 = calculate_activation_statistics(torch.cat(self.fake_activations, dim=0))
        m2, s2 = calculate_activation_statistics(torch.cat(self.real_activations, dim=0))
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

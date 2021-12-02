import torch
import torch.nn.functional as F

from sick import SimpleDICK

test_registry = []


def test(f):
    test_registry.append(f)
    return f


@test
def separable_equals_full():
    module = SimpleDICK(3)
    tensor = torch.randn(2, 1, 32, 32)
    x, _ = module(tensor)
    full_kernel = torch.outer(module.vertical_kernel, module.horizontal_kernel)
    x_full = F.conv2d(tensor, full_kernel.unsqueeze(0).unsqueeze(0), padding=module.kernel_size // 2)
    assert torch.allclose(x, x_full, atol=1e-7)

@test
def log_det_is_finite():
    repetitions = 100
    module = SimpleDICK(3)
    for dimension in (2 ** i for i in range(4, 10)):
        for r in range(repetitions):
            tensor = torch.randn(2, 1, dimension, dimension)
            _, log_det = module(tensor)
            # if not torch.isfinite(log_det).all():
            #     print(dimension, r, log_det.item())
            assert torch.isfinite(log_det).all()


if __name__ == "__main__":
    for test in test_registry:
        test()

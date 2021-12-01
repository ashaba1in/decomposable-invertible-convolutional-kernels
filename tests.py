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
    tensor = torch.randn(1, 1, 5, 5)
    x, log_det = module(tensor)

    full_kernel = torch.outer(module.vertical_kernel, module.horizontal_kernel)
    x_full = F.conv2d(tensor, full_kernel.unsqueeze(0).unsqueeze(0), padding=module.kernel_size // 2)
    assert torch.allclose(x, x_full, atol=1e-15)


if __name__ == "__main__":
    for test in test_registry:
        test()

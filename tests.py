import torch
import torch.nn.functional as F

from sick import SimpleDICK

test_registry = []


def test(reps=1):
    def test_wrapper(func):
        test_registry.append((func, reps))
        return func

    return test_wrapper


@test(reps=100)
def separable_equals_full_stress():
    module = SimpleDICK(3)
    tensor = torch.randn(2, 1, 32, 32)
    x, _ = module(tensor)
    full_kernel = torch.outer(module.vertical_kernel, module.horizontal_kernel)
    x_full = F.conv2d(tensor, full_kernel.unsqueeze(0).unsqueeze(0), padding=module.kernel_size // 2)
    assert x.shape == x_full.shape == tensor.shape
    assert torch.allclose(x, x_full, atol=1e-5)


@test(reps=100)
def log_det_is_finite_stress():
    for dimension in (2 ** i for i in range(4, 10)):
        module = SimpleDICK(3)
        tensor = torch.randn(2, 1, dimension, dimension)
        _, log_det = module(tensor)
        assert torch.isfinite(log_det).all()


# @test(reps=100)
def log_det_is_correct_stress():
    module = SimpleDICK(3)
    module.requires_grad_(False)
    h, w = 32, 32
    tensor = torch.randn(h * w, requires_grad=True)
    x, log_det = module(tensor.view(1, 1, h, w))
    x_flat = x.view(h * w)
    jac = torch.zeros(h * w, h * w)
    for i in range(h * w):
        grad_outputs = torch.zeros_like(x_flat)
        grad_outputs[i] = 1.0
        jac[i] = torch.autograd.grad(x_flat, tensor, grad_outputs=grad_outputs, retain_graph=True)[0]
    # jac = torch.autograd.functional.jacobian(lambda t: module(t.view(1, 1, h, w))[0].view(h * w), tensor)
    _, log_det2 = torch.linalg.slogdet(jac)
    print(log_det, log_det2)
    # assert torch.allclose(log_det, log_det2)


@test(reps=1)
def log_det_total_inaccuracy():
    diff = 0
    for _ in range(100):
        module = SimpleDICK(3)
        module.requires_grad_(False)
        h, w = 32, 32
        tensor = torch.randn(h * w, requires_grad=True)
        x, log_det = module(tensor.view(1, 1, h, w))
        x_flat = x.view(h * w)
        jac = torch.zeros(h * w, h * w)
        for i in range(h * w):
            grad_outputs = torch.zeros_like(x_flat)
            grad_outputs[i] = 1.0
            jac[i] = torch.autograd.grad(x_flat, tensor, grad_outputs=grad_outputs, retain_graph=True)[0]
        # jac = torch.autograd.functional.jacobian(lambda t: module(t.view(1, 1, h, w))[0].view(h * w), tensor)
        _, log_det2 = torch.linalg.slogdet(jac)
        diff += torch.abs(log_det - log_det2)
    print(diff / 100)


if __name__ == "__main__":
    for test, repetitions in test_registry:
        for rep in range(repetitions):
            test()

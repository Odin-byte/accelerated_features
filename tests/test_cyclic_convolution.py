import pytest
import torch
from modules.cyclic_convolution import CyclicConvolution


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_cyclic_convolution(device):
    device = torch.device(device)
    batch_size, channels, height, width = 1, 3, 32, 32
    kernel_size = 3

    model = CyclicConvolution(stride=1, padding=1).to(device)
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    kernel = torch.nn.Parameter(torch.randn(1, channels, kernel_size, kernel_size, device=device))

    output = model(input_tensor, kernel)

    assert output.shape == (batch_size, 1, height, width)
    assert output.device == device

    output.sum().backward()
    assert kernel.grad is not None
    assert kernel.grad.shape == kernel.shape

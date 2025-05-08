import torch
import torch.nn as nn
from modules.polar_rotation import PolarRotationTransform
from modules.cyclic_convolution import CyclicConvolution

# filepath: /home/dominik/PHD/code/xfeat/code/accelerated_features/modules/cyclic_polar_coordinate_conv.py


class CyclicPolarCoordinateConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Custom PyTorch layer that applies a polar rotation transform followed by a cyclic convolution.

        Args:
            kernel_size (int): Size of the learnable kernel for cyclic convolution.
            stride (int): Stride for the cyclic convolution. Default is 1.
        """
        super(CyclicPolarCoordinateConvLayer, self).__init__()
        self.polar_transform = PolarRotationTransform()
        self.cyclic_conv = CyclicConvolution()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define a learnable kernel for cyclic convolution
        self.kernel = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size, requires_grad=True)
        )

        self.cyclic_conv.set_stride(self.stride)
        self.cyclic_conv.set_padding(self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the custom layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying polar rotation and cyclic convolution.
        """
        # Ensure the kernel is on the same device as the input tensor
        self.kernel = self.kernel.to(x.device)

        # Apply polar rotation transform
        polar_transformed = self.polar_transform(x)

        # Perform cyclic convolution
        cyclic_conv = self.cyclic_conv.forward(polar_transformed, self.kernel)

        return cyclic_conv

if __name__ == "__main__":
    # Example usage
    batch_size, channels, height, width = 1, 3, 32, 32
    input_tensor = torch.rand((batch_size, channels, height, width))

    # Create an instance of the custom layer
    cyclic_polar_layer = CyclicPolarCoordinateConvLayer(kernel_size=3, stride=1, padding=1)

    # Forward pass
    output_tensor = cyclic_polar_layer(input_tensor)

    # Print the shape of the output tensor
    print("Output Tensor Shape:", output_tensor.shape)
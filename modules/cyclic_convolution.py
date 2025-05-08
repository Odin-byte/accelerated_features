import torch
import torch.nn as nn
import torch.nn.functional as F


class CyclicConvolution(nn.Module):
    def __init__(self, stride=1, padding=0):
        """Initialize the CyclicConvolution layer.

        Args:
            stride (int): Stride for the convolution operation.
            padding (int): Padding for the convolution operation.
        """
        super(CyclicConvolution, self).__init__()
        self.stride = stride
        self.padding = padding

    def set_stride(self, stride: int):
        """Set the stride for the convolution operation.
        The stride should be a positive integer.

        Args:
            stride (int): Stride value.
        """
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("Stride must be a positive integer.")
        self.stride = stride

    def set_padding(self, padding: int):
        """Set the padding for the convolution operation.
        Zero padding is used if given a positive integer.

        Args:
            padding (int): Length of padding added to each dimension.
        """
        if not isinstance(padding, int) or padding < 0:
            raise ValueError("Padding must be a positive integer.")
        self.padding = padding

    def forward(self, image: torch.Tensor, kernel: torch.nn.Parameter) -> torch.Tensor:
        """Perform the cyclic convolution operation.

        Args:
            image (torch.Tensor): Input tensor.
            kernel (torch.nn.Parameter): Convolution kernel.

        Returns:
            torch.Tensor: Output tensor after cyclic convolution.
        """
        # Ensure the input and kernel are on the same device
        device = image.device
        kernel = kernel.to(device)

        # Concat the input along the width dimension / side by side
        input_concat = torch.cat((image, image), 3)
        # Run the cyclic convolution
        output = F.conv2d(
            input_concat, kernel, padding=self.padding, stride=self.stride
        )
        return output


if __name__ == "__main__":
    # Generate a toy example using an input of 3x3 with the values 1,2,3,4,5,6,7,8,9
    # Create a 3x3 input tensor
    batch_size, channels, height, width = 1, 3, 24, 32
    input_tensor = torch.rand((batch_size, channels, height, width))

    test_layer = CyclicConvolution()
    kernel = nn.Parameter(torch.randn(1, 3, 3, 3, requires_grad=True))

    test_layer.set_padding(1)
    test_output = test_layer(input_tensor, kernel)
    print(test_output.size())

import torch
import torch.nn as nn
import torch.nn.functional as F


class CyclicConvolution(nn.Module):
    def __init__(self, stride=1):
        """Initialize the CyclicConvolution layer.

        Args:
            stride (int): Stride for the convolution operation.
        """
        super(CyclicConvolution, self).__init__()
        self.stride = stride

    def set_stride(self, stride: int):
        """Set the stride for the convolution operation.
        The stride should be a positive integer.

        Args:
            stride (int): Stride value.
        """
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("Stride must be a positive integer.")
        self.stride = stride

    def forward(self, image: torch.Tensor, kernel: torch.nn.Parameter) -> torch.Tensor:
        """Perform the cyclic convolution operation.

        Args:
            image (torch.Tensor): Input tensor.
            kernel (torch.nn.Parameter): Convolution kernel.

        Returns:
            torch.Tensor: Output tensor after cyclic convolution.
        """
        device = image.device
        kernel = kernel.to(device)

        kernel_height, kernel_width = kernel.size(2), kernel.size(3)
        pad_w = kernel_width // 2
        pad_h = kernel_height // 2

        # 1. Apply circular padding along width (left-right)
        image = F.pad(image, (pad_w, pad_w, 0, 0), mode='circular')  # pads last dim (W)

        # 2. Apply zero padding along height (top-bottom)
        image = F.pad(image, (0, 0, pad_h, pad_h), mode='constant', value=0)  # pads height

        # Now convolve
        output = F.conv2d(image, kernel, stride=self.stride, padding=0)
        return output


if __name__ == "__main__":
    # Create a 3x3 input tensor
    batch_size, channels, height, width = 1, 3, 377, 608
    # Generate a toy example using an input of 3x3 with the values 1,2,3,4,5,6,7,8,9
    input_tensor = torch.tensor([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]], dtype=torch.float32)
    # Make the tensor 4D by adding batch and channel dimensions
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1)

    test_layer = CyclicConvolution()
    kernel = nn.Parameter(torch.randn(1, 3, 3, 3, requires_grad=True))

    test_output = test_layer(input_tensor, kernel)

    print(input_tensor)
    print(test_output)
    print(test_output.size())

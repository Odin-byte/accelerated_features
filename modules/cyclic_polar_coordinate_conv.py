import torch
import cv2
import torch.nn as nn
from modules.polar_rotation import PolarRotationTransform
from modules.cyclic_convolution import CyclicConvolution
import torchvision.transforms as transforms

# filepath: /home/dominik/PHD/code/xfeat/code/accelerated_features/modules/cyclic_polar_coordinate_conv.py


class CyclicPolarCoordinateConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
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

        # Define a learnable kernel for cyclic convolution
        self.kernel = nn.Parameter(
            torch.randn(in_channels, out_channels, kernel_size, kernel_size, requires_grad=True)
        )

        self.cyclic_conv.set_stride(self.stride)

    def forward(self, x: torch.Tensor, center=None) -> torch.Tensor:
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
        polar_transformed = self.polar_transform(x, center)

        # Perform cyclic convolution
        cyclic_conv = self.cyclic_conv.forward(polar_transformed, self.kernel)

        return cyclic_conv

if __name__ == "__main__":
    # Load in a sample    # Read the image
    image = cv2.imread('/home/docker/torch/code/accelerated_features/modules/test_data/sunflower_rect.png')

    # Convert BGR image to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_LINEAR)

    # Define a transform to convert
    # the image to torch tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Rotate the image by 90 degrees
    image_rot = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)

    # Convert the image to Torch tensor
    tensor = transform(resized_image)
    images = tensor.unsqueeze(0)

    tensor_rotated = transform(image_rot
                               ).unsqueeze(0)

    # Create an instance of the custom layer
    cyclic_polar_layer = CyclicPolarCoordinateConvLayer(in_channels=3, out_channels=3, kernel_size=3, stride=1)


    H, W = resized_image.shape[:2]
    H = H//2
    print(H, W)
    original_center = (100, 100)

    # After 90 deg clockwise rotation
    rotated_center = (original_center[1] + H, original_center[0])  # (cy, W - 1 - cx)

    print(rotated_center)

    # Show both images with the corresponding center
    cv2.drawMarker(resized_image, original_center, color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.imshow("Normal Image", resized_image)
    cv2.drawMarker(image_rot, rotated_center, color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.imshow("Rotated Image", image_rot)

    # Forward pass
    output_tensor = cyclic_polar_layer(images, center=(100, 100))

    output_tensor_rotated = cyclic_polar_layer(tensor_rotated, center=rotated_center)
    
    # Print the shape of the output tensor
    print("Output Tensor Shape:", output_tensor.shape)
    print("Output Tensor Rotated Shape:", output_tensor_rotated.shape)

    # Show the output tensor
    output_image = output_tensor.squeeze(0).permute(1, 2, 0).detach().numpy()
    output_image_rotated = output_tensor_rotated.squeeze(0).permute(1, 2, 0).detach().numpy()

    # Convert the output tensor to an image format
    output_image = (output_image * 255).astype('uint8')
    output_image_rotated = (output_image_rotated * 255).astype('uint8')

    # Display the output images
    cv2.imshow('Cyclic Polar Coordinate Convolution Output', output_image)
    cv2.imshow('Cyclic Polar Coordinate Convolution Output Rotated', output_image_rotated)

    # Wait for a key press and close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Print the shape of the output tensor
    # print("Output Tensor Shape:", output_tensor.shape)
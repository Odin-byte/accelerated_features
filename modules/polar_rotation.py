import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


class PolarRotationTransform():
    def __init__(self):
        self.output_image = None

    def transform(self, image: torch.Tensor) -> torch.Tensor:
        """Apply a polar rotation transform to the input image. 
        The input and output images are PyTorch tensors. 
        The input image is transformed into a polar representation. 
        The output image has 3/4 of the input height while preserving the width.

        Args:
            image (torch.Tensor): Input image tensor of shape (N, C, H, W), 
                                  where N is the batch size,
                                  C is the number of channels, 
                                  H is the height, and W is the width.

        Returns:
            torch.Tensor: Transformed image tensor of shape (N, C, 3/4*H, W).
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a PyTorch tensor.")
        
        if image.dim() != 4:
            raise ValueError("Input image must have 4 dimensions (N, C, H, W).")
        
        N, C, H, W = image.shape
        R = H * 0.7  # Sample radius for polar coordinates

        # Create an empty output image tensor of the input image size
        self.output_image = torch.zeros((N, C, H, W), dtype=image.dtype, device=image.device)

        # Algorithm
        for n in range(N):  # Iterate over the batch
            for p in range(W):
                theta = 2 * math.pi * p / W
                for q in range(H):
                    r = q * R / H
                    x = int(r * math.cos(theta)) + W // 2
                    y = int(r * math.sin(theta)) + H // 2
                    if 0 <= x < W and 0 <= y < H:
                        self.output_image[n, :, q, p] = image[n, :, y, x]
                    else:
                        self.output_image[n, :, q, p] = 255

        # Resize the output image to 3/4 of the original height using nearest neighbor interpolation
        new_height = int(3 * H / 4)
        self.output_image = F.interpolate(self.output_image, size=(new_height, W), mode='nearest')
        
        return self.output_image
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Make the class callable."""
        return self.transform(image)
    
    # Example usage

if __name__ == "__main__":
    # Create a random batch of images tensor of shape (N, C, H, W)
    N, C, H, W = 4, 3, 32, 32
    images = torch.rand((N, C, H, W))

    # # Show one of the images
    # plt.imshow(images[0].permute(1, 2, 0).numpy())
    # plt.axis('off')
    # plt.show()

    # Create an instance of the PolarRotationTransform class
    polar_transform = PolarRotationTransform()

    # Apply the polar rotation transform
    transformed_images = polar_transform(images)

    # Print the shape of the transformed images
    print(transformed_images.shape)  # Expected output: (N, C, 192, W)

    # # Show one of the transformed images
    # plt.imshow(transformed_images[0].permute(1, 2, 0).numpy())
    # plt.axis('off')
    # plt.show()

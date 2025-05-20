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
    
    def polar_transform_vectorized(self, image):
        N, C, H, W = image.shape
        R = H * 0.7

        q = torch.arange(H, device=image.device).view(H, 1)
        p = torch.arange(W, device=image.device).view(1, W)

        r = q * R / H
        theta = 2 * math.pi * p / W

        x = (r * torch.cos(theta)).round().long() + W // 2
        y = (r * torch.sin(theta)).round().long() + H // 2

        in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)

        x_clamped = x.clamp(0, W - 1)
        y_clamped = y.clamp(0, H - 1)

        # Flatten indices for advanced indexing
        x_flat = x_clamped.view(-1)  # (H*W,)
        y_flat = y_clamped.view(-1)  # (H*W,)
        in_bounds_flat = in_bounds.view(-1)  # (H*W,)

        output = torch.full((N, C, H, W), 255, dtype=image.dtype, device=image.device)

        for n in range(N):
            # Gather pixels for batch n using flattened indices
            pixels = image[n, :, y_flat, x_flat]  # [C, H*W]

            # Set out-of-bounds pixels to 255 manually
            pixels[:, ~in_bounds_flat] = 255

            # Reshape back to [C, H, W]
            output[n] = pixels.view(C, H, W)

        new_height = int(3 * H / 4)
        output = F.interpolate(output, size=(new_height, W), mode='nearest')

        return output

    
    def test_polar_sample(self):
        torch.manual_seed(0)
        input = torch.rand(1, 3, 5, 5)

        output_loop = self.transform(input)
        output_vector = self.polar_transform(input)

        print(output_loop.shape)
        print(output_vector.shape)

        diff = (output_loop - output_vector).abs().max().item()
        print(f"Max difference: {diff:.6f}")
        assert diff < 1e-6, "Mismatch between loop and vectorized output!"

        print("✅ Test passed! Vectorized version matches loop exactly.")


    def cartesian_to_polar(self, x, y, H, W, R=None):
        """
        Map image coordinates (x, y) to polar feature map coordinates (q, p).
        Args:
            x: image x coordinate (can be tensor or scalar)
            y: image y coordinate (can be tensor or scalar)
            H: input image height
            W: input image width
            R: maximum radius (default: H * 0.7)
        Returns:
            q: radial index (float)
            p: angular index (float)
        """
        if R is None:
            R = H * 0.7
        dx = torch.tensor(x - W / 2, dtype=torch.float32)
        dy = torch.tensor(y - H / 2, dtype=torch.float32)
        r = torch.sqrt(dx ** 2 + dy ** 2)
        theta = torch.atan2(dy, dx) % (2 * math.pi)
        q = r * H / R
        p = theta * W / (2 * math.pi)
        return q, p
    
    def polar_to_cartesian(self, q, p, H, W, R=None):
        """
        Map polar feature map coordinates (q, p) to image coordinates (x, y).
        Args:
            q: radial index (can be tensor or scalar)
            p: angular index (can be tensor or scalar)
            H: input image height
            W: input image width
            R: maximum radius (default: H * 0.7)
        Returns:
            x, y: image coordinates (float)
        """
        if R is None:
            R = H * 0.7
        r = q * R / H
        theta = 2 * math.pi * p / W
        x = r * torch.cos(theta) + W / 2
        y = r * torch.sin(theta) + H / 2
        return x, y

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Make the class callable."""
        return self.polar_transform_vectorized(image)
    
    # Example usage

if __name__ == "__main__":
    
    N, C, H, W = 1, 3, 32, 32
    images = torch.rand((N, C, H, W))
    print(images[0])

    # # Show one of the images
    # plt.imshow(images[0].permute(1, 2, 0).numpy())
    # plt.axis('off')
    # plt.show()

    # Create an instance of the PolarRotationTransform class
    polar_transform = PolarRotationTransform()

    # polar_transform.test_polar_sample()

    # Apply the polar rotation transform
    transformed_images = polar_transform(images)

    # Apply the second version
    transformed_image_new = polar_transform.transform(images)

    print(transformed_image_new.shape)

    # Pick a point in the polar feature map
    q, p = 10, 15  # radial and angular indices

    # Map to Cartesian coordinates
    x, y = polar_transform.polar_to_cartesian(torch.tensor(q), torch.tensor(p), H, W)
    x_idx = int(round(x.item()))
    y_idx = int(round(y.item()))

    # Check bounds
    if 0 <= x_idx < W and 0 <= y_idx < H:
        orig_value = images[0, 2, y_idx, x_idx]
        polar_value = transformed_image_new[0, 2, q, p]
        print(f"Polar value: {polar_value}, Original value: {orig_value}")
    else:
        print("Mapped Cartesian coordinates are out of bounds.")

    # Reverse mapping check
    q_new, p_new = polar_transform.cartesian_to_polar(x_idx, y_idx, H, W)
    print(f"Reversed polar coordinates: q={q_new}, p={p_new}")
    # Check if the polar coordinates are close to the original
    print(f"Value at original coordinates: {images[0, 2, y_idx, x_idx]}")
    print(f"Value at polar coordinates: {transformed_image_new[0, 2, int(q_new), int(p_new)]}")
    print(q_new, p_new, q, p)

    # Print the shape of the transformed images
    print(transformed_images.shape)  # Expected output: (N, C, 192, W)
    print(transformed_image_new.shape)  # Expected output: (N, C, H, W)

    # print(transformed_images[0])
    # print(transformed_image_new[0])

    # # Show the first transformed image of the batch using both methods
    # plt.subplot(1, 2, 1)
    # plt.imshow(transformed_images[0].permute(1, 2, 0).numpy())
    # plt.axis('off')
    # plt.title("Polar Rotation Transform")
    # plt.subplot(1, 2, 2)
    # plt.imshow(transformed_image_new[0].permute(1, 2, 0).numpy())
    # plt.axis('off')
    # plt.title("Cartesian to Polar Transform")
    # plt.show()

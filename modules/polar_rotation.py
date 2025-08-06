import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import cv2
import torchvision.transforms as transforms
import numpy as np


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

    def polar_rotation_grid(self, H, W, R=None, device="cpu"):
        if R is None:
            R = max(H, W)
        q = torch.arange(H, device=device).view(-1, 1)
        p = torch.arange(W, device=device).view(1, -1)
        r = q * R / max(H, W)
        theta = 2 * math.pi * p / W
        x = r * torch.cos(theta) + (W - 1) / 2
        y = r * torch.sin(theta) + (H - 1) / 2
        x_norm = 2 * (x / (W - 1)) - 1
        y_norm = 2 * (y / (H - 1)) - 1
        grid = torch.stack((x_norm, y_norm), dim=-1)
        return grid
    
    def cartesian_to_polar_grid(self, H, W, R=None, device="cpu"):
        if R is None:
            R = max(H, W)
        y = torch.arange(H, device=device).view(-1, 1).expand(H, W)
        x = torch.arange(W, device=device).view(1, -1).expand(H, W)

        cx = (W - 1) / 2
        cy = (H - 1) / 2

        x_c = x - cx
        y_c = y - cy

        r = torch.sqrt(x_c ** 2 + y_c ** 2)
        theta = torch.atan2(y_c, x_c)
        theta = (theta + 2 * math.pi) % (2 * math.pi)

        q = r * max(H, W) / R
        p = theta * W / (2 * math.pi)

        # Normalize to [-1, 1]
        q_norm = 2 * (q / (H - 1)) - 1
        p_norm = 2 * (p / (W - 1)) - 1

        grid = torch.stack((p_norm, q_norm), dim=-1)  # (H, W, 2)
        return grid
    def polar_rotation_transform(self, image):
        # image: (N, C, H, W)
        N, C, H, W = image.shape
        grid = self.polar_rotation_grid(H, W, device=image.device)  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, H, W, 2)
        output = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return output
    
    def restore_cartesian_img_grid_sample(self, polar_img, H, W, R=None):
        device = polar_img.device
        grid = self.cartesian_to_polar_grid(H, W, R, device=device)
        grid = grid.unsqueeze(0).repeat(polar_img.shape[0], 1, 1, 1)  # (N, H, W, 2)
        # Note: grid_sample expects (N, C, H, W) and grid (N, H, W, 2)
        restored = F.grid_sample(polar_img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return restored
    
    def cartesian_to_polar_indices(self, x, y, H, W, R=None):
        """
        Accepts scalars or tensors for x, y.
        Returns q, p (can be tensors or scalars depending on input).
        """
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)
        
        device = x.device
        dtype = x.dtype

        if R is None:
            R = (max(H,W)-1)

        cx = (W - 1) / 2
        cy = (H - 1) / 2
        x_c = x - cx
        y_c = y - cy
        r = torch.sqrt(x_c**2 + y_c**2)
        theta = torch.atan2(y_c, x_c)
        theta = (theta + 2 * torch.pi) % (2 * torch.pi)
        q = r * max(H, W) / R
        p = theta * W / (2 * torch.pi)
        return p, q

    def polar_indices_to_cartesian(self, p, q, H, W, R=None):
        """
        Accepts scalars or tensors for p, q.
        Returns x, y (same shape and type as inputs).
        """
        p = torch.as_tensor(p)
        q = torch.as_tensor(q)

        device = p.device
        dtype = p.dtype

        if R is None:
            R = (max(H, W)-1)
        cx = (W - 1) / 2
        cy = (H - 1) / 2

        r = q * R / max(H, W)
        theta = p * 2 * torch.pi / W

        x = r * torch.cos(theta) + cx
        y = r * torch.sin(theta) + cy

        return x, y


    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Make the class callable."""
        return self.polar_rotation_transform(image)
    
def cartesian_to_polar_indices_tensor(xy_grid: torch.Tensor, H: int, W: int, R=None):
        """
        Args:
            xy_grid: Tensor of shape (N, H, W, 2) or (H, W, 2) with coordinates in (x, y) format.
            H: Height of image
            W: Width of image
            R: Optional max radius; defaults to min(H, W)

        Returns:
            polar_indices: Tensor of shape (..., 2) with (p, q) indices
        """
        if R is None:
            R = min(H, W)

        # Center coordinates
        cx = W / 2.0
        cy = H / 2.0

        x = xy_grid[..., 0] - cx  # shape (...,)
        y = xy_grid[..., 1] - cy

        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)
        theta = torch.remainder(theta, 2 * math.pi)  # Ensures [0, 2pi)

        q = r * H / R
        p = theta * W / (2 * math.pi)

        return torch.stack((p, q), dim=-1)
    # Example usage

if __name__ == "__main__":
    
    # N, C, H, W = 1, 3, 64, 64
    # images = torch.rand((N, C, H, W))

    # Read the image
    image = cv2.imread('/home/docker/torch/code/accelerated_features/modules/test_data/sunflower_rect.png')

    # Convert BGR image to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Move the image 100 Pixels to the right and 100 Pixels down
    image_moved = cv2.warpAffine(image,
                                  M=np.float32([[1, 0, 100], [0, 1, 100]]),
                                  dsize=(image.shape[1], image.shape[0]))

    # Define a transform to convert
    # the image to torch tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Convert the image to Torch tensor
    tensor = transform(image)

    images = tensor.unsqueeze(0)

    # Create an instance of the PolarRotationTransform class
    polar_transform = PolarRotationTransform()

    original_height, original_width = images.shape[2], images.shape[3]

    transformed_image_second_new = polar_transform.polar_rotation_transform(images)

    # Print the shape of the transformed image
    print("Input Image Shape:", images.shape)
    print("Transformed Image Shape:", transformed_image_second_new.shape)

    restored_img = polar_transform.restore_cartesian_img_grid_sample(transformed_image_second_new, original_height, original_width)


    # Check if the pixel which is at cx-10, cy-10 in the original image has the same value in the polar image
    cx, cy = original_width // 2, original_height // 2
    x, y = float(cx - 325), float(cy)

    x_y_grid = torch.tensor([x, y])
    
    # Get polar indices
    
    q, p = cartesian_to_polar_indices_tensor(x_y_grid, original_height, original_width)    

    # Show the first transformed image of the batch using both methods
    plt.subplot(1, 2, 1)
    cart_img_np = images[0].permute(1, 2, 0).numpy()
    plt.imshow(cart_img_np)
    plt.scatter(x, y, c="green")
    plt.axis('off')
    plt.title("Input")
    plt.subplot(1, 2, 2)
    polar_img_np = transformed_image_second_new[0].permute(1, 2, 0).numpy()
    plt.imshow(polar_img_np)
    plt.scatter(q, p, c="green")
    plt.axis('off')
    plt.title("Polar")

    plt.show()

    p, q = polar_transform.cartesian_to_polar_indices(x, y, original_height, original_width)
    x2, y2 = polar_transform.polar_indices_to_cartesian(p, q, original_height, original_width)

    print(f"Original: ({x}, {y}) → Polar: ({p}, {q}) → Back: ({x2}, {y2})")

    pixel_cart = cart_img_np[int(x), int(y)]
    pixel_polar = polar_img_np[int(p), int(q)]

    print(f"Diff: {pixel_cart-pixel_polar}")


    # Show the first transformed image of the batch using both methods
    plt.subplot(1, 4, 1)
    plt.imshow(images[0].permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title("Input")
    plt.subplot(1, 4, 2)
    plt.imshow(transformed_image_second_new[0].permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title("Polar")
    plt.subplot(1, 4, 3)
    # Calculate the difference image
    diff_img = torch.abs(images[0] - restored_img[0])
    diff = diff_img / diff_img.max()  # normalize for visualization
    plt.imshow(diff_img.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title("Difference")
    plt.subplot(1, 4, 4)
    plt.imshow(restored_img[0].permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title("Restored")

    plt.show()



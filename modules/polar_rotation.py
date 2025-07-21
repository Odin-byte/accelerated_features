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
    
    # def polar_transform_vectorized(self, image):
    #     N, C, H, W = image.shape
    #     R = H * 0.7

    #     q = torch.arange(H, device=image.device).view(H, 1)
    #     p = torch.arange(W, device=image.device).view(1, W)

    #     r = q * R / H
    #     theta = 2 * math.pi * p / W

    #     x = (r * torch.cos(theta)).round().long() + W // 2
    #     y = (r * torch.sin(theta)).round().long() + H // 2

    #     in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    #     x_clamped = x.clamp(0, W - 1)
    #     y_clamped = y.clamp(0, H - 1)

    #     # Flatten indices for advanced indexing
    #     x_flat = x_clamped.view(-1)  # (H*W,)
    #     y_flat = y_clamped.view(-1)  # (H*W,)
    #     in_bounds_flat = in_bounds.view(-1)  # (H*W,)

    #     output = torch.full((N, C, H, W), 255, dtype=image.dtype, device=image.device)

    #     for n in range(N):
    #         # Gather pixels for batch n using flattened indices
    #         pixels = image[n, :, y_flat, x_flat]  # [C, H*W]

    #         # Set out-of-bounds pixels to 255 manually
    #         pixels[:, ~in_bounds_flat] = 255

    #         # Reshape back to [C, H, W]
    #         output[n] = pixels.view(C, H, W)

    #     new_height = int(3 * H / 4)
    #     output = F.interpolate(output, size=(new_height, W), mode='nearest')

    #     return output
    

    def polar_rotation_grid(self, H, W, R=None, device="cpu"):
        if R is None:
            R = min(H, W)
        q = torch.arange(H, device=device).view(-1, 1)
        p = torch.arange(W, device=device).view(1, -1)
        r = q * R / H
        theta = 2 * math.pi * p / W
        x = r * torch.cos(theta) + W / 2
        y = r * torch.sin(theta) + H / 2
        # Normalize to [-1, 1] for grid_sample
        x_norm = 2 * (x / (W - 1)) - 1
        y_norm = 2 * (y / (H - 1)) - 1
        grid = torch.stack((x_norm, y_norm), dim=-1)  # (H, W, 2)
        return grid
    
    def cartesian_to_polar_grid(self, H, W, R=None, device="cpu"):
        if R is None:
            R = min(H, W)
        y = torch.arange(H, device=device).view(-1, 1).expand(H, W)
        x = torch.arange(W, device=device).view(1, -1).expand(H, W)
        x_c = x - W / 2
        y_c = y - H / 2
        r = torch.sqrt(x_c ** 2 + y_c ** 2)
        theta = torch.atan2(y_c, x_c)
        theta = (theta + 2 * math.pi) % (2 * math.pi)  # [0, 2pi]
        q = r * H / R
        p = theta * W / (2 * math.pi)
        # Normalize to [-1, 1]
        q_norm = 2 * (q / (H - 1)) - 1
        p_norm = 2 * (p / (W - 1)) - 1
        grid = torch.stack((p_norm, q_norm), dim=-1)  # (H, W, 2)
        return grid
    
    # # grid: (H_out, W_out, 2)
    # def get_nearest_indices(self, grid, H, W):
    #     x = ((grid[..., 0] + 1) * (W - 1) / 2)
    #     y = ((grid[..., 1] + 1) * (H - 1) / 2)
    #     x_idx = torch.round(x).long().clamp(0, W - 1)
    #     y_idx = torch.round(y).long().clamp(0, H - 1)
    #     return x_idx, y_idx  # Each (H_out, W_out)
    
    # def get_bilinear_indices_and_weights(self, grid, H, W):
    #     # grid: (N, H_out, W_out, 2), values in [-1, 1]
    #     # Convert normalized grid to image coordinates
    #     x = ((grid[..., 0] + 1) * (W - 1) / 2)
    #     y = ((grid[..., 1] + 1) * (H - 1) / 2)

    #     x0 = torch.floor(x).long()
    #     x1 = x0 + 1
    #     y0 = torch.floor(y).long()
    #     y1 = y0 + 1

    #     # Clip to valid range
    #     x0 = x0.clamp(0, W - 1)
    #     x1 = x1.clamp(0, W - 1)
    #     y0 = y0.clamp(0, H - 1)
    #     y1 = y1.clamp(0, H - 1)

    #     # Weights
    #     wx = x - x0.float()
    #     wy = y - y0.float()

    #     w00 = (1 - wx) * (1 - wy)
    #     w01 = wx * (1 - wy)
    #     w10 = (1 - wx) * wy
    #     w11 = wx * wy

    #     # Each of these is (N, H_out, W_out)
    #     return (x0, x1, y0, y1), (w00, w01, w10, w11)

    def polar_rotation_transform(self, image):
        # image: (N, C, H, W)
        N, C, H, W = image.shape
        grid = self.polar_rotation_grid(H, W, device=image.device)  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, H, W, 2)
        output = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return output
    
    # def restore_cartesian_img(self, cyclic_img, indices, weights, height, width):
    #     """
    #     Args:
    #         cyclic_img: (N, C, H_out, W_out)
    #         indices: (x0, x1, y0, y1) each (H_out, W_out)
    #         weights: (w00, w01, w10, w11) each (H_out, W_out)
    #         height, width: target cartesian image size
    #     Returns:
    #         cartesian_img: (N, C, height, width)
    #     """
    #     N, C, H_out, W_out = cyclic_img.shape
    #     cartesian_img = torch.zeros((N, C, height, width), dtype=cyclic_img.dtype, device=cyclic_img.device)
    #     x0, x1, y0, y1 = indices
    #     w00, w01, w10, w11 = weights

    #     for n in range(N):
    #         for c in range(C):
    #             # Flatten all indices and weights
    #             vals = cyclic_img[n, c].reshape(-1)  # (H_out*W_out,)
    #             for dx, dy, w in [
    #                 (x0, y0, w00), (x1, y0, w01), (x0, y1, w10), (x1, y1, w11)
    #             ]:
    #                 # dx, dy, w: (H_out, W_out)
    #                 dx_flat = dx.reshape(-1)
    #                 dy_flat = dy.reshape(-1)
    #                 w_flat = w.reshape(-1)
    #                 # Scatter-add: add weighted values to the corresponding cartesian pixels
    #                 cartesian_img[n, c].index_put_(
    #                     (dy_flat, dx_flat),
    #                     vals * w_flat,
    #                     accumulate=True
    #                 )
    #     return cartesian_img
    
    def restore_cartesian_img_grid_sample(self, polar_img, H, W, R=None):
        device = polar_img.device
        grid = self.cartesian_to_polar_grid(H, W, R, device=device)
        grid = grid.unsqueeze(0).repeat(polar_img.shape[0], 1, 1, 1)  # (N, H, W, 2)
        # Note: grid_sample expects (N, C, H, W) and grid (N, H, W, 2)
        restored = F.grid_sample(polar_img, grid, mode='bilinear', padding_mode='border', align_corners=True)
        return restored
    
    def cartesian_to_polar_indices(self, x, y, H, W, R=None):
        if R is None:
            R = min(H, W)
        cx = W / 2
        cy = H / 2

        x_c = x - cx
        y_c = y - cy

        r = math.sqrt(x_c**2 + y_c**2)
        theta = math.atan2(y_c, x_c)
        if theta < 0:
            theta += 2 * math.pi

        q = r * H / R
        p = theta * W / (2 * math.pi)

        return q, p
    
    def polar_indices_to_cartesian(self, p, q, H, W, R=None):
        if R is None:
            R = min(H, W)
        cx = W / 2
        cy = H / 2

        r = q * R / H
        theta = p * 2 * math.pi / W

        x = r * math.cos(theta) + cx
        y = r * math.sin(theta) + cy

        return x, y


    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Make the class callable."""
        return self.polar_rotation_transform(image)
    
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

    tensor_moved = transform(image_moved)

    images = tensor.unsqueeze(0)
    images_moved = tensor_moved.unsqueeze(0)

    # Show the point which is at cx-10, cy-10 in the original image
    cx, cy = images.shape[3] // 2, images.shape[2] // 2
    x, y = cx - 10, cy - 10

    # # Draw a red circle at the point (cx-10, cy-10) in the original image
    # cv2.circle(image, (x, y), radius=5, color=(255, 0, 0), thickness=-1)
    # cv2.circle(image_moved, (x + 200, y + 100), radius=5, color=(255, 0, 0), thickness=-1)

    # # Show the images with the point marked
    # plt.figure(figsize=(10, 5))
    
    # # Show the original image with the point marked
    # plt.subplot(1, 2, 1)
    # plt.imshow(image)
    # plt.axis('off')
    # plt.title("Original Image with Point")

    # # Show the moved image with the point marked
    # plt.subplot(1, 2, 2)
    # plt.imshow(image_moved)
    # plt.axis('off')
    # plt.title("Moved Image with Point")
    
    # plt.show()

    # Create an instance of the PolarRotationTransform class
    polar_transform = PolarRotationTransform()

    # Apply the polar rotation transform
    # transformed_images = polar_transform(images)

    # # Apply the second version
    # transformed_image_new = polar_transform.transform(images)

    original_height, original_width = images.shape[2], images.shape[3]

    transformed_image_second_new = polar_transform.polar_rotation_transform(images)

    # Print the shape of the transformed image
    print("Input Image Shape:", images.shape)
    print("Transformed Image Shape:", transformed_image_second_new.shape)

    restored_img = polar_transform.restore_cartesian_img_grid_sample(transformed_image_second_new, original_height, original_width)

    transformed_image_second_new_moved, _, _ = polar_transform.polar_rotation_transform(images_moved)


    # Check if the pixel which is at cx-10, cy-10 in the original image has the same value in the polar image
    cx, cy = original_width // 2, original_height // 2
    x, y = cx - 10, cy - 10
    
    # Get polar indices
    q, p = polar_transform.cartesian_to_polar_indices(x, y, original_height, original_width)
    q = int(round(q))
    p = int(round(p))
    # Check bounds
    if 0 <= q < transformed_image_second_new.shape[2] and 0 <= p < transformed_image_second_new.shape[3]:
        polar_value_before_moved = transformed_image_second_new[0, :, q, p]
    else:
        print("Mapped polar coordinates are out of bounds.")

    # Get the corresponding pixel in the moved image
    x_moved, y_moved = x + 100, y + 100
    # Get polar indices for the moved image
    q_moved, p_moved = polar_transform.cartesian_to_polar_indices(x_moved, y_moved, original_height, original_width)
    q_moved = int(round(q_moved))
    p_moved = int(round(p_moved))
    
    # Check bounds for the moved image
    if 0 <= q_moved < transformed_image_second_new.shape[2] and 0 <= p_moved < transformed_image_second_new.shape[3]:
        polar_value_after_moved = transformed_image_second_new_moved[0, :, q_moved, p_moved]
    else:
        print("Mapped polar coordinates for moved image are out of bounds.")

    print(f"Polar value before moving: {polar_value_before_moved}")
    print(f"Polar value after moving: {polar_value_after_moved}")

    # Calculate the euclidean distance between the two polar values
    distance = torch.norm(polar_value_before_moved - polar_value_after_moved)
    print(f"Euclidean distance between polar values: {distance.item()}")

    # Check if coordinate transform is reversible
    x = 10
    y = 10
    q, p = polar_transform.cartesian_to_polar_indices(x, y, original_height
, original_width)
    x_restored, y_restored = polar_transform.polar_indices_to_cartesian(p, q, original_height, original_width)

    print(f"Original coordinates: ({x}, {y})")
    print(f"Polar indices: (q={round(q)}, p={round(p)})")
    print(f"Restored coordinates: ({round(x_restored)}, {round(y_restored)})")


    

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

"""This code defines functions for performing interpolation of 3D data in polar coordinates."""

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from modules.cyclic_polar_patches import PolarPatchGenerator
from modules.polar_rotation import PolarRotationTransform

def norm_grid_to_patch_at_center(norm_grid, patch_size, center_x, center_y):
    half = (patch_size - 1) / 2.0
    out = norm_grid.clone()
    out[..., 0] = (norm_grid[..., 0] + 1) / 2 * (patch_size - 1) - half + center_x
    out[..., 1] = (norm_grid[..., 1] + 1) / 2 * (patch_size - 1) - half + center_y
    return out

# Load in a test image
test_image_path = '/home/docker/torch/code/accelerated_features/modules/test_data/rainbow.png'

# Convert BGR image to RGB image
image = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)

# Define a transform to convert
# the image to torch tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# Convert the image to Torch tensor
test_tensor = transform(image)

patch_size = 129
# Create three patches. One at center, one shifted right, one shifted left
patch_gen = PolarPatchGenerator(3, 3, patch_size, 3, 1)

image_height, image_width = test_tensor.shape[1], test_tensor.shape[2]
center_x, center_y = image_width // 2, image_height // 2
points = torch.tensor([[[center_x, center_y],
                        [center_x + patch_size//2, center_y],
                        [center_x - patch_size//2, center_y]]], dtype=torch.float32)  # Shape (1, 3, 2)

polar_patches = patch_gen.forward(test_tensor.unsqueeze(0), points)
print("Extracted polar patches shape:", polar_patches.shape)

# Interpolate the center patch based on the left and right patch
polar_transform_layer = PolarRotationTransform()
# Generate the grid for interpolation that would be used for the center patch
grid_global_img = polar_transform_layer.polar_rotation_grid(patch_size, patch_size, R=(patch_size - 1) / 2)
# Convert the grid to image coordinates
grid_global_img = norm_grid_to_patch_at_center(grid_global_img, patch_size, center_x, center_y)
# Shift the grid to be centered at the image center
# grid_global_img[..., 0] += center_x
# grid_global_img[..., 1] += center_y
# Show the global grid points on the image
grid_global_img_np = grid_global_img.numpy().astype(np.int32)
image_with_grid = image.copy()
for i in range(patch_size):
    for j in range(patch_size):
        x, y = grid_global_img_np[i, j]
        if 0 <= x < image_width and 0 <= y < image_height:
            image_with_grid[y, x] = [0, 0, 0]  # Mark grid points in black
cv2.imshow("Image with Polar Grid", image_with_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Visualize the patches
import matplotlib.pyplot as plt
def visualize_patches(patches):
    B, N, C, H, W = patches.shape
    for n in range(N):
        patch = patches[0, n].permute(1, 2, 0).detach().numpy()  # Convert to HWC
        plt.subplot(1, N, n + 1)
        plt.imshow(patch)
        plt.axis('off')
    plt.show()

visualize_patches(polar_patches)

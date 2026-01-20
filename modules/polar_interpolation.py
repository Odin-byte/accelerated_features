"""This code defines functions for performing interpolation of 3D data in polar coordinates."""

import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from modules.cyclic_polar_patches import PolarPatchGenerator
from modules.polar_rotation import PolarRotationTransform


def interpolate_polar_patch_from_other_patches(source_patches, number_of_patches_per_row,
                                              number_of_patches_per_col, patch_size, target_x, target_y):
    """
    Interpolate a patch at a given cartesian coordinate from surrounding polar patches.
    """
    # Restore the cartesian representation of the source patches
    B, N, C, H, W = source_patches.shape
    assert N == number_of_patches_per_row * number_of_patches_per_col, "Number of source patches does not match the specified grid dimensions."
    polar_transform_layer = PolarRotationTransform()
    restored_patches = []
    for i in range(N):
        patch = source_patches[0, i]  # (C, H, W)
        restored_patch = polar_transform_layer.restore_cartesian_img_grid_sample(patch.unsqueeze(0), patch_size, patch_size)
        restored_patches.append(restored_patch)
    restored_patches = torch.cat(restored_patches, dim=0)  # (N, C, H, W)

    # Generate grid of patch center coordinates
    # Interpolate the center patch based on the left and right patch
    polar_transform_layer = PolarRotationTransform()
    # Generate the grid for interpolation that would be used for the center patch
    grid_global_img = polar_transform_layer.polar_rotation_grid(patch_size, patch_size)
    # Convert the grid to pixel coordinates relative to the entire image
    # Stich together all patches into a big image
    big_image_height = number_of_patches_per_col * patch_size
    big_image_width = number_of_patches_per_row * patch_size
    grid_global_img = norm_grid_to_patch_at_center(grid_global_img, patch_size, target_x, target_y, big_image_width, big_image_height)
    big_image = torch.zeros((3, big_image_height, big_image_width), device=restored_patches.device)
    for i in range(number_of_patches_per_col):
        for j in range(number_of_patches_per_row):
            patch_index = i * number_of_patches_per_row + j
            patch = restored_patches[patch_index]  # (C, H, W)
            start_y = i * patch_size
            start_x = j * patch_size
            big_image[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = patch
    # Show the big image
    import matplotlib.pyplot as plt
    plt.imshow(big_image.permute(1, 2, 0).detach().numpy())
    plt.axis('off')
    plt.show()

    # Sample the interpolated patch from the big image
    big_image = big_image.unsqueeze(0)  # (1, C, H, W)
    interpolated_patch = torch.nn.functional.grid_sample(big_image, grid_global_img.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True)
    return interpolated_patch  # (1, C, H, W)

def interpolate_polar_patch_from_other_patches_in_polar(source_patches, number_of_patches_per_row,
                                              number_of_patches_per_col, patch_size, target_x, target_y):
    """
    Interpolate a patch at a given coordinate by directly sampling from polar patches.
    """
    B, N, C, H, W = source_patches.shape
    polar_transform_layer = PolarRotationTransform()
    
    # Generate the polar grid for the target patch
    target_polar_grid = polar_transform_layer.polar_rotation_grid(patch_size, patch_size)
    
    # For each point in the target polar grid, determine which source patch to sample from
    # Convert polar coordinates to global Cartesian coordinates
    big_image_height = number_of_patches_per_col * patch_size
    big_image_width = number_of_patches_per_row * patch_size
    
    # Get global Cartesian coordinates for the target polar grid
    global_cart_grid = norm_grid_to_patch_at_center(
        target_polar_grid, patch_size, target_x, target_y, 
        big_image_width, big_image_height
    )
    
    # Convert to pixel coordinates to determine which patch each point belongs to
    px = (global_cart_grid[..., 0] + 1) / 2 * (big_image_width - 1)
    py = (global_cart_grid[..., 1] + 1) / 2 * (big_image_height - 1)
    
    # Determine which patch each point belongs to
    patch_col = (px / patch_size).long().clamp(0, number_of_patches_per_row - 1)
    patch_row = (py / patch_size).long().clamp(0, number_of_patches_per_col - 1)
    
    # Get local coordinates within each source patch (Cartesian)
    local_x = px - patch_col.float() * patch_size
    local_y = py - patch_row.float() * patch_size
    
    # Convert local Cartesian to local polar coordinates for each source patch
    polar_transform = PolarRotationTransform()
    local_polar_grid = torch.stack([local_x, local_y], dim=-1)
    
    # Convert to polar indices in source patches
    local_polar_p, local_polar_q = polar_transform.cartesian_to_polar_indices(
        local_x, local_y, patch_size, patch_size
    )
    
    # Sample directly from polar patches
    interpolated_patch = torch.zeros((1, C, patch_size, patch_size), device=source_patches.device)

    for i in range(patch_size):
        for j in range(patch_size):
            row = patch_row[i, j].item()
            col = patch_col[i, j].item()
            patch_idx = row * number_of_patches_per_row + col
            
            p = local_polar_p[i, j].item()
            q = local_polar_q[i, j].item()
            
            # Clamp q (radius) to valid range
            q = max(0, min(q, patch_size - 1))
            
            # Wrap p (theta) to handle cyclic boundary
            p = p % patch_size
            
            # Get integer parts and fractional parts
            q0 = int(q)
            p0 = int(p)
            q1 = min(q0 + 1, patch_size - 1)
            p1 = (p0 + 1) % patch_size  # Cyclic wrapping
            
            dq = q - q0
            dp = p - p0
            
            # Bilinear interpolation with proper bounds checking
            v00 = source_patches[0, patch_idx, :, q0, p0]
            v01 = source_patches[0, patch_idx, :, q0, p1]
            v10 = source_patches[0, patch_idx, :, q1, p0]
            v11 = source_patches[0, patch_idx, :, q1, p1]
            
            interpolated_patch[0, :, i, j] = (
                v00 * (1 - dq) * (1 - dp) +
                v01 * (1 - dq) * dp +
                v10 * dq * (1 - dp) +
                v11 * dq * dp
            )

    return interpolated_patch
    

def norm_grid_to_patch_at_center(norm_grid, patch_size, center_x, center_y, image_width, image_height):
    half = (patch_size - 1) / 2.0
    # patch-local pixels around the target center
    px = (norm_grid[..., 0] + 1) / 2 * (patch_size - 1) - half + center_x
    py = (norm_grid[..., 1] + 1) / 2 * (patch_size - 1) - half + center_y
    # normalize to full-image [-1, 1] for grid_sample
    out = norm_grid.new_empty(norm_grid.shape)
    out[..., 0] = 2 * (px / (image_width - 1)) - 1
    out[..., 1] = 2 * (py / (image_height - 1)) - 1
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

patch_size = 55
# Create three patches. One at center, one shifted right, one shifted left
patch_gen = PolarPatchGenerator(3, 3, patch_size, 3, 1)

image_height, image_width = test_tensor.shape[1], test_tensor.shape[2]
print("Image size:", image_height, image_width)
# Generate a list of patch center points with size 129 so that the entire image is covnered with patches without overlapping the single patches
num_patches_per_row = image_width // patch_size
num_patches_per_col = image_height // patch_size
print("Number of patches per row:", num_patches_per_row)
print("Number of patches per column:", num_patches_per_col)

# Resize the image to num_patches_per_row * patch_size x num_patches_per_col * patch_size
resized_width = num_patches_per_row * patch_size
resized_height = num_patches_per_col * patch_size
test_tensor = transforms.functional.resize(test_tensor, (resized_height, resized_width))
print("Resized image size:", test_tensor.shape[1], test_tensor.shape[2])

points = []
for i in range(num_patches_per_col):
    for j in range(num_patches_per_row):
        center_x = j * patch_size + patch_size // 2
        center_y = i * patch_size + patch_size // 2
        points.append([center_x, center_y])
points = torch.tensor(points).unsqueeze(0)  # Shape (1, N, 2)

polar_patches = patch_gen.forward(test_tensor.unsqueeze(0), points)
print("Extracted polar patches shape:", polar_patches.shape)

# Interpolate the patch at the center of the image from surrounding patches
target_center_x = resized_width // 2 +10
target_center_y = resized_height // 2 +10
interpolated_patch = interpolate_polar_patch_from_other_patches_in_polar(polar_patches, num_patches_per_row,
                                                      num_patches_per_col, patch_size,
                                                      target_center_x, target_center_y).unsqueeze(0)

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

# Generate a real center patch for comparison
real_center_patch = patch_gen.forward(test_tensor.unsqueeze(0),
                                     torch.tensor([[[resized_width // 2 + 10, resized_height // 2 + 10]]]))

print("Real center patch shape:", real_center_patch.shape)
print("Interpolated patch shape:", interpolated_patch.shape)

# Combine the real and interpolated patches for visualization as 1, 2, C, H, W
patches_to_compare = torch.cat([real_center_patch, interpolated_patch], dim=1)

visualize_patches(patches_to_compare)
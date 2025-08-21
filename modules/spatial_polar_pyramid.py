import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision.transforms as transforms
from torchvision.transforms.functional import gaussian_blur
import numpy as np
from modules.cyclic_polar_coordinate_conv import CyclicPolarCoordinateConvLayer

class SpatialPolarPyramid(nn.Module):
    def __init__(self, in_channels, out_channels, num_levels=4, kernel_size=3):
        super().__init__()
        self.num_levels = num_levels
        self.convs = nn.ModuleList([
            # Create an instance of the custom layer
            CyclicPolarCoordinateConvLayer(in_channels=1, out_channels=1, kernel_size=3, stride=1)
            for _ in range(num_levels)
        ])

    def split_and_process(self, x, overlap_ratio=0.5):
        N, C, H, W = x.shape
        level_tensors = []
        for level, conv in enumerate(self.convs):
            splits = 2 ** level
            patch_H, patch_W = H // splits, W // splits
            stride = (int(patch_H * (1-overlap_ratio)), int(patch_W * (1-overlap_ratio)))

            # Collect patches for all images in the batch
            all_patches = []
            for n in range(N):
                patches = F.unfold(x[n:n+1], kernel_size=(patch_H, patch_W), stride=stride)
                num_patches = patches.shape[-1]
                patches = patches.transpose(1, 2).reshape(num_patches, C, patch_H, patch_W)
                all_patches.append(patches)  # (num_patches, channels, patch_H, patch_W)

            # Concatenate all patches from all images: (N * num_patches, channels, patch_H, patch_W)
            all_patches = torch.cat(all_patches, dim=0)
            # Now process all patches in one batch
            processed = conv(all_patches)  # (N * num_patches, out_channels, patch_H, patch_W)
            level_tensors.append(processed)
        return level_tensors  # List of tensors, one per level

    def forward(self, x):
        return self.split_and_process(x)
    

class PolarPatchStitching(nn.Module):
    def __init__(self, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.batch_size = 2  # Set this dynamically if needed
        self.cyclic_conv = CyclicPolarCoordinateConvLayer(in_channels=1, out_channels=1, kernel_size=3, stride=1)

    def stitch_patches(self, image_patches, overlap_ratio=0.5, apply_gaussian_blur=False, kernel_size=(7, 7), sigma=(1.5, 1.5)):
        # image_patches: (num_patches, channels, patch_H, patch_W)
        num_patches = image_patches.shape[0]
        channels = image_patches.shape[1]
        patch_H = image_patches.shape[2]
        patch_W = image_patches.shape[3]
        grid_size = int(torch.sqrt(torch.tensor(num_patches, dtype=torch.float32)))
        assert grid_size * grid_size == num_patches, "num_patches must be a perfect square"

        # Set stride for overlap
        stride_H = int(patch_H * (1 - overlap_ratio))
        stride_W = int(patch_W * (1 - overlap_ratio))
        stride_H = max(1, stride_H)
        stride_W = max(1, stride_W)

        # Prepare for fold
        patches = image_patches.reshape(num_patches, channels * patch_H * patch_W)
        patches = patches.transpose(0, 1).unsqueeze(0)  # (1, channels * patch_H * patch_W, num_patches)

        # Calculate output size
        out_H = stride_H * (grid_size - 1) + patch_H
        out_W = stride_W * (grid_size - 1) + patch_W

        # Fold to reconstruct
        stitched = F.fold(
            patches,
            output_size=(out_H, out_W),
            kernel_size=(patch_H, patch_W),
            stride=(stride_H, stride_W)
        )  # (1, channels, H, W)

        # Create count map for averaging overlaps
        ones = torch.ones((1, channels, out_H, out_W), dtype=stitched.dtype, device=stitched.device)
        ones_patches = F.unfold(ones, kernel_size=(patch_H, patch_W), stride=(stride_H, stride_W))
        count_map = F.fold(
            ones_patches,
            output_size=(out_H, out_W),
            kernel_size=(patch_H, patch_W),
            stride=(stride_H, stride_W)
        )
        stitched = stitched / count_map.clamp(min=1e-6)  # Avoid division by zero

        if apply_gaussian_blur:
            stitched = gaussian_blur(stitched, kernel_size=kernel_size, sigma=sigma)

        return stitched.squeeze(0)  # (channels, H, W)

    def forward(self, level_tensors: list[torch.Tensor]):
        restored_levels = []
        for level_tensor in level_tensors:
            num_patches = level_tensor.shape[0] // self.batch_size
            channels = level_tensor.shape[1]
            patch_H = level_tensor.shape[2]
            patch_W = level_tensor.shape[3]

            # Batch restore: pass all patches at once
            restored = self.cyclic_conv.polar_transform.restore_cartesian_img_grid_sample(
                level_tensor, patch_H, patch_W
            )  # Should support batch input

            # Unflatten back to (batch_size, num_patches, channels, patch_H, patch_W)
            restored_unflattened = restored.view(self.batch_size, num_patches, channels, patch_H, patch_W)
            restored_levels.append(restored_unflattened)

        # For every batch item, collect stitched images for each level
        batch_stitched = []
        for b in range(self.batch_size):
            stitched_per_level = []
            for level in restored_levels:
                stitched_img = self.stitch_patches(level[b])  # (channels, H, W)
                stitched_per_level.append(stitched_img)
            # Concatenate levels along channel dimension: (num_levels * channels, H, W)
            stitched_concat = torch.cat(stitched_per_level, dim=0)
            batch_stitched.append(stitched_concat.unsqueeze(0))  # (1, num_levels * channels, H, W)

        # Stack to get (batch_size, num_levels * channels, H, W)
        output = torch.cat(batch_stitched, dim=0)
        return output


if __name__ == "__main__":
    # Example usage
    polar_pyramid_network = SpatialPolarPyramid(1, 1)
    
    # Load test img
    image = cv2.imread('/home/docker/torch/code/accelerated_features/modules/test_data/sunflower_rect.png')
    # Convert BGR image to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make it greyscale with one channel
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Crop the image to the next size divisible by 2
    h, w = image.shape
    factor = 2 ** 4
    new_h = (h // factor) * factor
    new_w = (w // factor) * factor
    image = image[:new_h, :new_w]

    image = np.expand_dims(image, axis=-1)

    # Define a transform to convert
    # the image to torch tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Convert the image to Torch tensor
    tensor = transform(image)
    # Flip the tensor around by 180 degrees
    flipped_tensor = tensor.flip(-1)
    # Add the flipped tensor as a second input along the batch dimension
    input_tensor = torch.cat([tensor.unsqueeze(0), flipped_tensor.unsqueeze(0)], dim=0)



    patches = polar_pyramid_network(input_tensor)

    stitching_layer = PolarPatchStitching(num_levels=4)
    stitched_images = stitching_layer.forward(patches)
    print(stitched_images.shape)  # Should be (batch_size, channels, H, W)

    # Show the second channel of the first image in my stitched images tensor
    second_channel = stitched_images[1][3]
    print(second_channel.shape)  # Should be (H, W)
    second_channel = second_channel.detach().unsqueeze(0).numpy()
    second_channel = np.transpose(second_channel, (1, 2, 0))
    # Convert to uint8
    second_channel = (second_channel * 255).astype(np.uint8)
    cv2.imshow("Second Channel", second_channel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
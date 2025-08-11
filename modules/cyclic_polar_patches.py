import torch
import cv2
import torch.nn as nn
import kornia
from modules.cyclic_convolution import CyclicConvolution
import torchvision.transforms as transforms

# filepath: /home/dominik/PHD/code/xfeat/code/accelerated_features/modules/cyclic_polar_coordinate_conv.py


class PolarPatchGenerator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        """
        Custom PyTorch layer that applies a patch-based polar rotation transform followed by a cyclic convolution for every point coordinate given.

        Args:
            kernel_size (int): Size of the learnable kernel for cyclic convolution.
            stride (int): Stride for the cyclic convolution. Default is 1.
        """
        super(PolarPatchGenerator, self).__init__()
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

    def extract_centered_patches(self, feats, kp_px, patch_size):
        B, C, H, W = feats.shape
        half = patch_size // 2

        patches = []
        for b in range(B):
            b_patches = []
            for (x, y) in kp_px[b]:
                x, y = int(x.item()), int(y.item())
                x1, x2 = x - half, x + half
                y1, y2 = y - half, y + half
                patch = feats[b, :, y1:y2, x1:x2]  # shape (C, H, W)
                b_patches.append(patch)
            b_patches = torch.stack(b_patches, dim=0)  # (N, C, patch_size, patch_size)
            patches.append(b_patches)
        return torch.stack(patches, dim=0)  # (B, N, C, patch_size, patch_size)

    def forward(self, feats: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the custom layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying polar rotation and cyclic convolution.
        """
        # Ensure the kernel is on the same device as the input tensor
        self.kernel = self.kernel.to(feats.device)

        B, C, H, W = feats.shape  # feats from CNN
        N = keypoints.shape[1]    # number of keypoints

        patches = self.extract_centered_patches(feats, keypoints, 32)

        # Perform cyclic convolution
        B, N, C, H, W = patches.shape
        print(C)
        patches_reshaped = patches.view(B * N, C, H, W)
        cyclic_conv_patches = self.cyclic_conv.forward(patches_reshaped, self.kernel)
        # Restore grouping into (B, N, ...)
        cyclic_conv_patches = cyclic_conv_patches.view(B, N, cyclic_conv_patches.shape[1],
                                                    cyclic_conv_patches.shape[2],
                                                    cyclic_conv_patches.shape[3])

        return cyclic_conv_patches

if __name__ == "__main__":
    # Load in a sample    # Read the image
    image = cv2.imread('/home/docker/torch/code/accelerated_features/modules/test_data/sunflower_rect.png')

    # Convert BGR image to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define a transform to convert
    # the image to torch tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Convert the image to Torch tensor
    tensor = transform(image)
    images = tensor.unsqueeze(0)

    test_keypoints = torch.Tensor([[300, 300]])
    test_keypoints = test_keypoints.unsqueeze(0)
    print(test_keypoints.shape)


    polar_patches = PolarPatchGenerator(3, 3, 3)
    polar_patches = polar_patches.forward(images, test_keypoints)
    print(polar_patches.shape)


    patches = polar_patches.squeeze(0)
    print(patches.shape)
    print(patches[0].shape)

    img = patches[0].detach().cpu().numpy()  # Convert to numpy
    print(img.shape)
    img = img.transpose(1, 2, 0)             # C,H,W -> H,W,C
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Normalize img to 0-255 uint8 for visualization
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype('uint8')

    # If single channel, convert grayscale to BGR
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imshow("Patch 0", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
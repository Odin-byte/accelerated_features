import torch
from modules.model import XFeatModel

def test_xfeat_model_output_sizes():
    """
    Test the output sizes of the XFeatModel.
    """
    # Define input dimensions
    batch_size = 4
    channels = 1  # Grayscale input
    height = 128
    width = 128

    # Create a random input tensor
    input_tensor = torch.rand((batch_size, channels, height, width))

    # Initialize the model
    model = XFeatModel()

    # Forward pass
    feats, keypoints, heatmap = model(input_tensor)

    # Expected output dimensions
    expected_feats_size = (batch_size, 64, height // 8, width // 8)
    expected_keypoints_size = (batch_size, 65, height // 8, width // 8)
    expected_heatmap_size = (batch_size, 1, height // 8, width // 8)

    # Assertions
    assert feats.size() == expected_feats_size, f"Expected feats size {expected_feats_size}, but got {feats.size()}"
    assert keypoints.size() == expected_keypoints_size, f"Expected keypoints size {expected_keypoints_size}, but got {keypoints.size()}"
    assert heatmap.size() == expected_heatmap_size, f"Expected heatmap size {expected_heatmap_size}, but got {heatmap.size()}"

    print("All output sizes are correct!")

if __name__ == "__main__":
    test_xfeat_model_output_sizes()
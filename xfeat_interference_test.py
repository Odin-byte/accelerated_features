import cv2
import time
import os
from modules.xfeat import XFeat
import torchvision.transforms as transforms

if __name__ == "__main__":


    # Load in a sample    # Read the image
    image = cv2.imread('/home/docker/torch/code/accelerated_features/modules/test_data/four.png')

    # Convert BGR image to RGB image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define a transform to convert
    # the image to torch tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Convert the image to Torch tensor
    tensor = transform(image)
    input_tensor = tensor.unsqueeze(0)

    # Initialize the XFeat model
    # weigh_path = os.path.abspath(os.path.dirname(__file__)) + "/weights/xfeat_cpcl.pth"
    model = XFeat()

    output = model.detectAndCompute(input_tensor)

    print(output[0]["keypoints"].shape)
    print(output[0]["scores"].shape)
    print(output[0]["descriptors"].shape)

    # Get the keypoint positions
    keypoints = output[0]["keypoints"].cpu().numpy()

    # Show the keypoints on the image
    for keypoint in keypoints:
        x, y = int(keypoint[0]), int(keypoint[1])
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
    # Display the image with keypoints
    cv2.imshow("Keypoints", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

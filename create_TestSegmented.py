import os
import cv2
import numpy as np
import torch
from packages.models.UNet import UNet

# Define paths
test_folder = "/mnt/2T/BreastCancerAll/.dataset/TESTSET/CM_TEST"
output_folder = "/mnt/2T/BreastCancerAll/.dataset/TESTSET/CMSEGMENTED_TEST"
model_path = "save:manual:unet[cm_mlo]_orgdata/best_val/dice/whole.pth"

# Initialize the segmentation model
segmentation_model = UNet(n_channels=3, n_classes=2, bilinear=True)

# Load the trained model weights
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
segmentation_model.load_state_dict(state_dict, strict=False)
segmentation_model.eval()


for root, dirs, files in os.walk(test_folder):
    for file in files:
        # Assuming the images are in a common image format (e.g., JPEG, PNG)
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, file)

            # Load the image
            original_image = cv2.imread(image_path)

            # Preprocess the image for the model
            input_image = torch.from_numpy(original_image).permute(2, 0, 1).float() / 255.0
            input_image = input_image.unsqueeze(0)

            # Make a prediction using the segmentation model
            with torch.no_grad():
                output = segmentation_model(input_image)

            # Assuming output is a tensor, extract the second channel as the mask
            mask = output[0, 1].numpy()

            # Ensure the mask has the same shape as the original image
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

            # Multiply the original image with the mask
            result_image = (original_image * mask[:,:,np.newaxis]).astype(np.uint8)

            # Save the result to the output folder
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, result_image)

print("Processing complete.")
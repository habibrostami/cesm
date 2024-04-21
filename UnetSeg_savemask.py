import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import numpy as np
from packages.models.UNet import UNet


# Load the pre-trained U-Net models
cmcc_model = UNet(3, 2, True)
cmcc_model.load_state_dict(torch.load('./save:manual:unet[cm_cc]/best_val/dice/whole.pth'))
cmcc_model.eval()

cmmlo_model = UNet(3, 2, True)
cmmlo_model.load_state_dict(torch.load('./save:manual:unet[cm_mlo]_500/best_val/dice/whole.pth'))
cmmlo_model.eval()

dmcc_model = UNet(3, 2, True)
dmcc_model.load_state_dict(torch.load('./save:manual:unet[dm_cc]_500/best_val/dice/whole.pth'))
dmcc_model.eval()

dmmlo_model = UNet(3, 2, True)
dmmlo_model.load_state_dict(torch.load('./save:manual:unet[dm_mlo]_500/best_val/dice/whole.pth'))
dmmlo_model.eval()

# Create a dictionary mapping image types to models
image_type_to_model = {
    'cm_cc': cmcc_model,
    'cm_mlo': cmmlo_model,
    'dm_cc': dmcc_model,
    'dm_mlo': dmmlo_model,
}

# Function to preprocess an image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust size as needed
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Function to predict mask and ROI
def predict_mask_and_roi(model, image):
    with torch.no_grad():
        # Forward pass to obtain the predicted mask
        output = model(image)
        predicted_mask = torch.argmax(output, dim=1).squeeze().numpy()

        # Convert the mask to binary (1 for ROI, 0 for background)
        roi = (predicted_mask == 1).astype(np.uint8)

    return predicted_mask, roi

# Function to save masks and ROIs to folders
def save_masks_and_rois_to_folders(image_paths, save_masks_folder, save_rois_folder, image_type_to_model):
    os.makedirs(save_masks_folder, exist_ok=True)
    os.makedirs(save_rois_folder, exist_ok=True)

    for image_path in image_paths:
        # Extract image type from the file name

        filename = os.path.basename(image_path)
        parts = filename.split('_')
        image_type = '_'.join(parts[-2:]).split('.')[0].lower()  # Take the last two parts and convert to uppercase

        # Preprocess the image
        image = preprocess_image(image_path)

        # Get the corresponding model for the image type
        model = image_type_to_model.get(image_type)
        if model is None:
            print(f"No model found for image type: {image_type}")
            continue

        # Predict mask and ROI
        predicted_mask, roi_mask = predict_mask_and_roi(model, image)

        # Convert torch tensor to NumPy array
        image_np = image.squeeze().permute(1, 2, 0).numpy()

        # Perform element-wise multiplication to obtain ROI
        # roi = (image_np * roi_mask[:, :, np.newaxis]).astype(np.uint8)
        roi = image_np * roi_mask[:, :, np.newaxis]

        # Save the mask
        mask_filename = os.path.join(save_masks_folder, f'mask_{filename}')
        cv2.imwrite(mask_filename, predicted_mask * 255)  # Assuming mask values are in the range [0, 1]

        # Save the ROI
        roi_filename = os.path.join(save_rois_folder, f'roi_{filename}')
        cv2.imwrite(roi_filename, roi)


# Provide a list of image paths and run the inference
image_folder = './data/DM/0'
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png'))]
save_masks_folder = '/mnt/2T/BreastCancerAll/.dataset/save_segmask_dm/0'
save_rois_folder = '/mnt/2T/BreastCancerAll/.dataset/save_segROI_dm/0'

save_masks_and_rois_to_folders(image_paths, save_masks_folder, save_rois_folder, image_type_to_model)

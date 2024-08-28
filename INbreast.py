import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from packages.models.UNet import UNet

# Initialize and load models
model_dm_cc = UNet(3, 2, True)
model_dm_cc.load_state_dict(torch.load('./save:manual:unet[dm_cc]_500/best_val/dice/whole.pth'))
model_dm_cc.eval()

model_dm_mlo = UNet(3, 2, True)
model_dm_mlo.load_state_dict(torch.load('./save:manual:unet[dm_mlo]_500/best_val/dice/whole.pth'))
model_dm_mlo.eval()

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, (256, 256))  # Resize to match the model's input size
    image = image.astype(np.float32) / 255.0  # Normalize the pixel values
    image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB by stacking
    return np.expand_dims(image, axis=0)  # Add batch dimension

def preprocess_image_for_model(image):
    image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
    image = image.permute(0, 3, 1, 2)  # Reorder dimensions to [batch_size, channels, height, width]
    return image

def visualize_results(original_image, mask_dm_cc=None, mask_dm_mlo=None):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.title('Original Image')

    if mask_dm_cc is not None:
        plt.subplot(1, 3, 2)
        plt.imshow(mask_dm_cc.squeeze(), cmap='gray')
        plt.title('DM_CC Segmentation')

    if mask_dm_mlo is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(mask_dm_mlo.squeeze(), cmap='gray')
        plt.title('DM_MLO Segmentation')

    plt.show()

inbreast_root = '/mnt/2T/BreastCancerAll/.dataset/INbreast/INbreast'  # Path to the INbreast dataset
output_root = '/mnt/2T/BreastCancerAll/.dataset/INbreast/INbreast_masks'  # Where you want to save the masks

for label in ['0', '1']:
    image_folder = os.path.join(inbreast_root, label)

    output_folder_dm_cc = os.path.join(output_root, 'dm_cc', label)
    output_folder_dm_mlo = os.path.join(output_root, 'dm_mlo', label)

    os.makedirs(output_folder_dm_cc, exist_ok=True)
    os.makedirs(output_folder_dm_mlo, exist_ok=True)

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)

        # Determine the view from the filename
        if 'CC' in image_name.upper():
            # Load and preprocess the CC image
            inbreast_image = load_and_preprocess_image(image_path)

            # Convert to tensor
            inbreast_image_tensor = preprocess_image_for_model(inbreast_image)

            # Apply the DM_CC model
            with torch.no_grad():
                mask_dm_cc = model_dm_cc(inbreast_image_tensor).cpu().numpy()

            # Check the output dimensions
            print(f"Mask for {image_name} shape before argmax: {mask_dm_cc.shape}")

            # Handle the multi-channel output
            if mask_dm_cc.ndim == 4 and mask_dm_cc.shape[1] == 2:
                # Apply argmax to convert to single channel (2D)
                mask_dm_cc = np.argmax(mask_dm_cc, axis=1)  # Take the argmax over the channel dimension

            # Ensure the mask is a 2D grayscale image
            if mask_dm_cc.ndim == 3 and mask_dm_cc.shape[0] == 1:
                mask_dm_cc = mask_dm_cc.squeeze(0)  # Remove the batch dimension if present

            if mask_dm_cc.ndim == 2:
                mask_dm_cc = (mask_dm_cc * 255).astype(np.uint8)
            else:
                print(f"Mask for {image_name} is not 2D after argmax. Shape: {mask_dm_cc.shape}")

            # Ensure mask is 2D before saving
            if mask_dm_cc.ndim == 2:
                # Save the mask
                cv2.imwrite(os.path.join(output_folder_dm_cc, image_name), mask_dm_cc)
            else:
                print(f"Mask for {image_name} is not 2D. Shape: {mask_dm_cc.shape}")

            # Optionally, visualize the result
            # visualize_results(inbreast_image.squeeze(), mask_dm_cc, None)

        elif 'MLO' in image_name.upper():
            # Load and preprocess the MLO image
            inbreast_image = load_and_preprocess_image(image_path)

            # Convert to tensor
            inbreast_image_tensor = preprocess_image_for_model(inbreast_image)

            # Apply the DM_MLO model
            with torch.no_grad():
                mask_dm_mlo = model_dm_mlo(inbreast_image_tensor).cpu().numpy()

            # Check the output dimensions
            print(f"Mask for {image_name} shape before argmax: {mask_dm_mlo.shape}")

            # Handle the multi-channel output
            if mask_dm_mlo.ndim == 4 and mask_dm_mlo.shape[1] == 2:
                # Apply argmax to convert to single channel (2D)
                mask_dm_mlo = np.argmax(mask_dm_mlo, axis=1)  # Take the argmax over the channel dimension

            # Ensure the mask is a 2D grayscale image
            if mask_dm_mlo.ndim == 3 and mask_dm_mlo.shape[0] == 1:
                mask_dm_mlo = mask_dm_mlo.squeeze(0)  # Remove the batch dimension if present

            if mask_dm_mlo.ndim == 2:
                mask_dm_mlo = (mask_dm_mlo * 255).astype(np.uint8)
            else:
                print(f"Mask for {image_name} is not 2D after argmax. Shape: {mask_dm_mlo.shape}")

            # Ensure mask is 2D before saving
            if mask_dm_mlo.ndim == 2:
                # Save the mask
                cv2.imwrite(os.path.join(output_folder_dm_mlo, image_name), mask_dm_mlo)
            else:
                print(f"Mask for {image_name} is not 2D. Shape: {mask_dm_mlo.shape}")

            # Optionally, visualize the result
            # visualize_results(inbreast_image.squeeze(), None, mask_dm_mlo)
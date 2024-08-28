import cv2
import numpy as np
import os

def highlight_and_save(original_image_path, mask_path, output_folder):
    # Load the original image and the mask (both should have the same dimensions)
    original_image = cv2.imread(original_image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize the mask to match the dimensions of the original image
    mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

    # Normalize the mask to values between 0 and 1
    normalized_mask = mask / 255.0

    # Define a subtle highlight color (e.g., light gray)
    highlight_color = np.array([255, 255, 255], dtype=np.uint8)

    # Apply the highlight to the original image where the mask is non-zero
    colored_highlight = original_image.copy()
    for i in range(3):  # Apply highlight to each channel (R, G, B)
        colored_highlight[:, :, i] = np.where(normalized_mask > 0,
                                              (highlight_color[i] * normalized_mask).astype(np.uint8),
                                              original_image[:, :, i])

    # Blend the original image with the colored highlight to create a subtle effect
    alpha = 0.5 # Control the intensity of the highlight
    highlighted_image = cv2.addWeighted(original_image, 1 - alpha, colored_highlight, alpha, 0)

    # Normalize the final image to be between 0 and 255
    normalized_image = cv2.normalize(highlighted_image, None, 0, 255, cv2.NORM_MINMAX)

    # Ensure the normalized image has the correct data type
    normalized_image = normalized_image.astype(np.uint8)

    # Save the result to the specified output folder
    output_path = os.path.join(output_folder, os.path.basename(original_image_path))
    cv2.imwrite(output_path, normalized_image)

# The root folder containing CM, DM, and INbreast folders
root_folder = '/mnt/2T/BreastCancerAll/.dataset/INbreast'

# Iterate through CM, DM, and INbreast folders
for folder_name in ['CM', 'DM', 'INbreast']:
    folder_path = os.path.join(root_folder, folder_name)

    if os.path.exists(folder_path):
        for label_folder_name in ['0', '1']:
            label_folder_path = os.path.join(folder_path, label_folder_name)
            if os.path.exists(label_folder_path):
                # Create folders to store highlighted images
                output_folder = os.path.join(root_folder, 'INbreast_UnetAOL', folder_name, label_folder_name)
                os.makedirs(output_folder, exist_ok=True)

                # Iterate through images in the label folder
                for image_file in os.listdir(label_folder_path):
                    if image_file.endswith('.jpg'):
                        original_image_path = os.path.join(label_folder_path, image_file)
                        mask_name = os.path.splitext(image_file)[0] + '.jpg'
                        mask_path = os.path.join(root_folder, 'INbreast_masks', label_folder_name, mask_name)

                        if os.path.exists(mask_path):
                            print(f"Processing: {original_image_path}")
                            highlight_and_save(original_image_path, mask_path, output_folder)

print("Highlighting, normalizing, and saving complete.")

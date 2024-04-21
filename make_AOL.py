
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

    # Define a highlight color (e.g., white)
    highlight_color = np.array([255, 255, 255], dtype=np.uint8)

    # Create a colored highlight image using the highlight color and the normalized mask
    colored_highlight = original_image.copy()
    colored_highlight[:, :] = original_image * normalized_mask[:, :, np.newaxis]

    # Ensure all images have the same data type
    colored_highlight = colored_highlight.astype(np.uint8)

    # Save the result to the specified output folder
    output_path = os.path.join(output_folder, os.path.basename(original_image_path))
    cv2.imwrite(output_path, colored_highlight)

# The rest of your code remains unchanged...


# Define the root folder containing CM and DM folders
root_folder = '/mnt/2T/BreastCancerAll/.dataset/root'

# Iterate through CM and DM folders
for folder_name in ['CM', 'DM']:
    folder_path = os.path.join(root_folder, folder_name)

    if os.path.exists(folder_path):
        for label_folder_name in ['0', '1']:
            label_folder_path = os.path.join(folder_path, label_folder_name)
            if os.path.exists(label_folder_path):
                # Create folders to store highlighted images
                output_folder = os.path.join(root_folder, 'highlighted_images', folder_name, label_folder_name)
                os.makedirs(output_folder, exist_ok=True)

                # Iterate through images in the label folder
                for image_file in os.listdir(label_folder_path):
                    if image_file.endswith('.jpg'):
                        original_image_path = os.path.join(label_folder_path, image_file)
                        mask_name = 'mask_' + os.path.splitext(image_file)[0] + '.jpg'
                        mask_path = os.path.join(root_folder, 'masks', mask_name)


                        if os.path.exists(mask_path):
                            print(f"Processing: {original_image_path}")
                            highlight_and_save(original_image_path, mask_path, output_folder)

print("Highlighting and saving complete.")


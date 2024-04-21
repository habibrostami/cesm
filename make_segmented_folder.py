import cv2
import os
import numpy as np
import torchvision.transforms as T
from torchvision import transforms
import torch
from PIL import Image

import os
from PIL import Image
import shutil

# Function to process an image using the provided code
def process_image(input_path, original_path):
    # Modify the paths in your code to use input_path and original_path
    img = Image.open(input_path)  # mask
    main_image = Image.open(original_path)

    convert_tensor = transforms.ToTensor()
    conv = convert_tensor(img)

    conv = torch.where(conv != 0, torch.tensor(1), conv)

    main_conv = convert_tensor(main_image)

    # Resize conv tensor to match the size of main_conv tensor
    conv = transforms.Resize(main_conv.shape[-2:])(conv)

    prepr = main_conv * conv

    transform_to_pil = T.ToPILImage()
    proc_img = transform_to_pil(prepr)
    # main_image.show()
    # proc_img.show()
    # img = transform_to_pil(conv)
    # img.show()

    # Save the processed image to a new folder
    output_folder = os.path.join('/mnt/2T/BreastCancerAll/.dataset/Unet_ROI', os.path.basename(os.path.dirname(original_path)))
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, os.path.basename(original_path))
    print(output_path)
    proc_img.save(output_path)

    return output_path

# Define the root folder containing CM and DM folders
root_folder = '/mnt/2T/BreastCancerAll/.dataset/root'
masks_folder = '/mnt/2T/BreastCancerAll/.dataset/save_Unetmasks/all'

# Iterate through CM and DM folders
for folder_name in ['CM', 'DM']:
    folder_path = os.path.join(root_folder, folder_name)

    if os.path.exists(folder_path):
        for label_folder_name in ['0', '1']:
            label_folder_path = os.path.join(folder_path, label_folder_name)
            if os.path.exists(label_folder_path):
                for image_file in os.listdir(label_folder_path):
                    if image_file.endswith('.jpg'):
                        input_image_path = os.path.join(label_folder_path, image_file)
                        mask_name = 'mask_' + os.path.splitext(image_file)[0] + '.jpg'
                        mask_path = os.path.join(masks_folder, mask_name)
                        print(mask_path)

                        if os.path.exists(mask_path):
                            print(mask_path)
                            processed_image_path = process_image(mask_path, input_image_path)
                            print(f"Processed: {input_image_path} -> {processed_image_path}")

print("Processing complete.")

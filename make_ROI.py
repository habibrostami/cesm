#

import cv2
import numpy as np
import os

def extract_and_save_roi(original_image_path, mask_path, output_folder):
    # Load the original image and the mask
    original_image = cv2.imread(original_image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize the mask to match the dimensions of the original image
    mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

    # Apply the mask to the original image (element-wise multiplication)
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Save the masked image as the ROI
    output_path = os.path.join(output_folder, os.path.basename(original_image_path))
    cv2.imwrite(output_path, masked_image)
# The root folder containing CM, DM, and INbreast folders
root_folder = '/mnt/2T/BreastCancerAll/.dataset/INbreast'

# Iterate through CM, DM, and INbreast folders
for folder_name in ['CM', 'DM', 'INbreast']:
    folder_path = os.path.join(root_folder, folder_name)

    if os.path.exists(folder_path):
        for label_folder_name in ['0', '1']:
            label_folder_path = os.path.join(folder_path, label_folder_name)
            if os.path.exists(label_folder_path):
                # Create folders to store ROI images
                output_folder = os.path.join(root_folder, 'INbreast_UnetROI', folder_name, label_folder_name)
                os.makedirs(output_folder, exist_ok=True)

                # Iterate through images in the label folder
                for image_file in os.listdir(label_folder_path):
                    if image_file.endswith('.jpg'):
                        original_image_path = os.path.join(label_folder_path, image_file)
                        mask_name = os.path.splitext(image_file)[0] + '.jpg'
                        mask_path = os.path.join(root_folder, 'INbreast_masks', label_folder_name, mask_name)

                        if os.path.exists(mask_path):
                            print(f"Processing ROI: {original_image_path}")
                            extract_and_save_roi(original_image_path, mask_path, output_folder)

print("ROI extraction and saving complete.")

# from PIL import Image
# import torch
# from torchvision import transforms
# import torchvision.transforms as T
#
# # Read image
#
# img = Image.open('./data/masks/P102_R_CM_CC.jpg.png') #mask
# main_image = Image.open('./data/org/CM/1/P102_R_CM_CC.jpg')
#
# convert_tensor = transforms.ToTensor()
# conv = convert_tensor(img)
#
#
# #remove heart
# #conv =  torch.where(conv == 1, torch.tensor(0), conv)
# conv = torch.where(conv != 0, torch.tensor(1), conv)
#
#
# main_conv = convert_tensor(main_image)
# prepr = main_conv * conv
#
# transform_to_pil = T.ToPILImage()
#
# proc_img = transform_to_pil(prepr)
# # main_image.show()
# proc_img.show()
# # Output Images
# img = transform_to_pil(conv)
# # img.show()
# print(type(img))
#
# import numpy as np
# np.savetxt('/stuff/saves_txt/my_file.txt', conv.numpy()[0])
#
# # prints format of image
# print(img.format)
#
# # prints mode of image
# print(img.mode)

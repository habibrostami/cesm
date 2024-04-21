import os
import shutil

# Define the root folder containing 0 and 1 label folders
root_folder = '/mnt/2T/BreastCancerAll/.dataset/Unet_ROI'

# Iterate through 0 and 1 label folders
for label_folder_name in ['0', '1']:
    label_folder_path = os.path.join(root_folder, label_folder_name)

    if os.path.exists(label_folder_path):
        # Create folders to store CM and DM images inside each label folder
        cm_folder_path = os.path.join(label_folder_path, 'CM')
        dm_folder_path = os.path.join(label_folder_path, 'DM')
        os.makedirs(cm_folder_path, exist_ok=True)
        os.makedirs(dm_folder_path, exist_ok=True)

        # Iterate through images in the label folder
        for image_file in os.listdir(label_folder_path):
            # print(image_file)
            if image_file.endswith('.jpg'):
                input_image_path = os.path.join(label_folder_path, image_file)

                # Check if the image is CM or DM based on the file name
                if 'CM' in image_file:
                    cm_output_path = os.path.join(cm_folder_path, image_file)
                    shutil.copy(input_image_path, cm_output_path)
                elif 'DM' in image_file:
                    dm_output_path = os.path.join(dm_folder_path, image_file)
                    shutil.copy(input_image_path, dm_output_path)

print("Separation complete.")

import os
import shutil

# Define the path to your main folder
main_folder = '/mnt/2T/BreastCancerAll/Base Code/data/reshaped_segmented'

# Define the paths for DM and CM folders
dm_folder = os.path.join(main_folder, 'DM')
cm_folder = os.path.join(main_folder, 'CM')

# Create DM and CM folders if they don't exist
for folder in [dm_folder, cm_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
    for label in ['0', '1']:
        subfolder = os.path.join(folder, label)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)


# Function to determine whether an image contains 'DM' or 'CM'
def is_dm_image(image_name):
    return 'DM' in image_name


def is_cm_image(image_name):
    return 'CM' in image_name


# Iterate through the '0' and '1' folders in the main directory
for label in ['0', '1']:
    label_folder = os.path.join(main_folder, label)
    for root, dirs, files in os.walk(label_folder):
        for file in files:
            if file.endswith('.jpg'):  # Change the file extension to the one you're using
                image_path = os.path.join(root, file)

                if is_dm_image(file):
                    destination_folder = os.path.join(dm_folder, label)
                elif is_cm_image(file):
                    destination_folder = os.path.join(cm_folder, label)
                else:
                    continue  # Skip images that don't contain 'DM' or 'CM'

                shutil.copy(image_path, os.path.join(destination_folder, file))

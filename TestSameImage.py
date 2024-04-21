import os
import shutil

# Define the paths to the DM and CM folders
dm_folder = '/mnt/2T/BreastCancerAll/.dataset/seg_preprocessed_images/DM'  # Replace with the actual path to your DM folder
cm_folder = '/mnt/2T/BreastCancerAll/.dataset/seg_preprocessed_images/CM'  # Replace with the actual path to your CM folder


# Define the output folder where missing images will be moved
target_folder = '/mnt/2T/BreastCancerAll/.dataset/seg_preprocessed_images/uncomplete_images'  # Replace with the actual path to the output folder


# Function to move missing images to the target folder
def move_missing_images(dm_folder, cm_folder, target_folder):
    for subfolder in ['0', '1']:
        dm_subfolder = os.path.join(dm_folder, subfolder)
        cm_subfolder = os.path.join(cm_folder, subfolder)
        target_subfolder = os.path.join(target_folder, subfolder)

        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder)

        for dm_image_name in os.listdir(dm_subfolder):
            cm_image_name = dm_image_name.replace('DM', 'CM')
            cm_image_path = os.path.join(cm_subfolder, cm_image_name)

            if not os.path.exists(cm_image_path):
                # Move the DM image to the target folder
                source_path = os.path.join(dm_subfolder, dm_image_name)
                target_path = os.path.join(target_subfolder, dm_image_name)
                shutil.move(source_path, target_path)
                print(f"Moved {dm_image_name} from {dm_subfolder} to {target_subfolder}")


# Move missing images to the target folder
move_missing_images(dm_folder, cm_folder, target_folder)
import os
import random
import shutil

# Define the paths to the DM and CM folders
dm_folder = '/mnt/2T/BreastCancerAll/.dataset/UNET_highlighted_images/DM'  # Replace with the actual path to your DM folder
cm_folder = '/mnt/2T/BreastCancerAll/.dataset/UNET_highlighted_images/all'  # Replace with the actual path to your CM folder

# Define the paths to the target folders
dm_target_folder = '/mnt/2T/BreastCancerAll/Results/data-images/Unet_AOL_split/DM'
cm_target_folder = '/mnt/2T/BreastCancerAll/Results/data-images/Unet_AOL_split/all'

# Define the percentages for test, validation, and train
test_percentage = 0.15
validation_percentage = 0.15
train_percentage = 0.7

# Function to partition and copy images
def partition_images(source_folder, target_folder):
    for subfolder in ['0', '1']:
        source_subfolder = os.path.join(source_folder, subfolder)
        target_train_subfolder = os.path.join(target_folder, 'train', subfolder)
        target_test_subfolder = os.path.join(target_folder, 'test', subfolder)
        target_validation_subfolder = os.path.join(target_folder, 'validation', subfolder)

        # Create train, test, and validation subfolders
        os.makedirs(target_train_subfolder, exist_ok=True)
        os.makedirs(target_test_subfolder, exist_ok=True)
        os.makedirs(target_validation_subfolder, exist_ok=True)

        image_names = os.listdir(source_subfolder)
        random.shuffle(image_names)

        num_images = len(image_names)
        num_test = int(num_images * test_percentage)
        num_validation = int(num_images * validation_percentage)
        num_train = num_images - num_test - num_validation

        # Copy images to train, test, and validation subfolders
        for i, image_name in enumerate(image_names):
            if i < num_test:
                shutil.copy(
                    os.path.join(source_subfolder, image_name),
                    os.path.join(target_test_subfolder, image_name)
                )
            elif i < num_test + num_validation:
                shutil.copy(
                    os.path.join(source_subfolder, image_name),
                    os.path.join(target_validation_subfolder, image_name)
                )
            else:
                shutil.copy(
                    os.path.join(source_subfolder, image_name),
                    os.path.join(target_train_subfolder, image_name)
                )

# Partition the DM folder
# partition_images(dm_folder, dm_target_folder)

# Partition the CM folder
partition_images(cm_folder, cm_target_folder)

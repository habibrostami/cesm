import cv2
import os
import numpy as np


def convert_to_minimal_squares(image_path, output_directory):
    # Read the mask image
    mask = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert the mask to a binary image using adaptive thresholding
    _, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_image = mask.copy()
    # Process each contour separately
    for i, contour in enumerate(contours):
        # Get the minimal bounding rectangle for each contour
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Create a mask for the bounding box
        square_mask = np.zeros_like(result_image)
        cv2.drawContours(square_mask, [box], 0, (255), thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255), 2)
        cv2.fillPoly(result_image, pts=[np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])], color=(255,255,255))


        # Save the resulting square mask to the output directory
        filename = os.path.join(output_directory, os.path.basename(image_path))
        cv2.imwrite(filename, result_image)


if __name__ == "__main__":

    input_directory = "/mnt/2T/BreastCancerAll/.dataset/masks"
    output_directory = "/mnt/2T/BreastCancerAll/.dataset/reshaped_masks"

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Process each mask in the input directory
    for mask_filename in os.listdir(input_directory):
        if mask_filename.endswith(".png"):
            mask_path = os.path.join(input_directory, mask_filename)
            convert_to_minimal_squares(mask_path, output_directory)



def find_tumors(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Convert the image to binary using adaptive thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours of connected components
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around tumors
    result_image = image.copy()
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (255), 2)

        cv2.fillPoly(result_image, pts=[np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])], color=(255,255,255))

    # Save the result
    cv2.imwrite(output_path, result_image)



if __name__ == "__main__":
    input_image_path = "/mnt/2T/BreastCancerAll/.dataset/masks/P2_R_CM_CC.jpg.png"
    output_image_path = "/mnt/2T/BreastCancerAll/.dataset/resulttP2_R_CM_CC.png"

    # find_tumors(input_image_path, output_image_path)










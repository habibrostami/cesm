import csv
import cv2
import random
import albumentations as A


def apply_augmentations(image, num_augmentations):
    augmentations = A.Compose([
        A.Rotate(limit=20, p=0.8),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(p=0.6),
        A.Lambda(image=random_zoom_image, p=0.7)
    ])

    augmented_images = [augmentations(image=image)['image'] for _ in range(num_augmentations)]
    return augmented_images


def random_zoom_image(image, **_):
    zoom_factor = random.uniform(1, 1.2)
    image_height, image_width = image.shape[:2]
    new_height, new_width = int(image_height * zoom_factor), int(image_width * zoom_factor)
    zoomed_image = cv2.resize(image, (new_width, new_height))
    return zoomed_image


def process_row(row, output_writer):
    x0, x1, x2, x3, y = row

    image_paths = [x0, x1, x2, x3]

    label = int(y)

    num_augmentations = 12 if label == 0 else 8

    augments = []
    for image_path in image_paths:
        image = cv2.imread(image_path)

        augmented_images = apply_augmentations(image, num_augmentations)
        augments.append([])

        for i, augmented_image in enumerate(augmented_images, start=1):
            image_name = f"{image_path.split('/')[-1].split('.')[0]}.aug{i}.jpg"
            output_path = f"{'/'.join(image_path.split('/')[:-1])}/{image_name}"
            augments[-1].append(output_path)
            cv2.imwrite(output_path, augmented_image)

    for a, b, c, d in zip(*augments):
        output_writer.writerow([a, b, c, d, y])


def main(input_csv_file, output_csv_file):
    with open(input_csv_file, 'r') as input_file, open(output_csv_file, 'w', newline='') as output_file:
        input_reader = csv.reader(input_file)
        output_writer = csv.writer(output_file)
        for i, row in enumerate(input_reader):
            print(f'\rRows processed: {i}', end='')
            process_row(row, output_writer)


if __name__ == "__main__":

    input_csv_file_path = "../New Code/train_data.csv"
    output_csv_file_path = "../New Code/train_augment.csv"
    main(input_csv_file_path, output_csv_file_path)

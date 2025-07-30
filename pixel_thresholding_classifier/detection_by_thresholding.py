from pathlib import Path

import cv2
from hyperparameters import threshold_value

from pixel_thresholding_classifier.test import TEST_PATH
from pixel_thresholding_classifier.train import TRAIN_PATH, VALIDATION_PATH

SAMPLES_PATH = Path.cwd() / "output" / "samples.csv"


def main():
    find_best_value(TRAIN_PATH)

    print(f"validation images: {count_mistakes(VALIDATION_PATH)}")

    print(f"test images: {count_mistakes(TEST_PATH)}")


def find_best_value(imgs_path):
    tagged_img_path = imgs_path / "tagged"
    untagged_img_path = imgs_path / "untagged"

    tagged_imgs = list(tagged_img_path.glob("*.png"))
    untagged_imgs = list(untagged_img_path.glob("*.png"))

    mistakes = []
    for value in range(256):
        mistake_count = 0
        for image_path in tagged_imgs + untagged_imgs:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            tagged = False
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i, j] > value:
                        tagged = True

            if tagged and "/tagged/" not in str(image_path):
                mistake_count += 1
                # print(f"FALSE POSITIVE - {image_path.name}")
            elif not tagged and "/untagged/" not in str(image_path):
                # print(f"FALSE NEGATIVE - {image_path.name}")
                mistake_count += 1
        mistakes.append((value, mistake_count))

    lowest = mistakes[0]
    for value, mistake_count in mistakes:
        if mistake_count < lowest[1]:
            lowest = (value, mistake_count)
    print(lowest)


def count_mistakes(imgs_path):
    tagged_img_path = imgs_path / "tagged"
    untagged_img_path = imgs_path / "untagged"

    tagged_imgs = list(tagged_img_path.glob("*.png"))
    untagged_imgs = list(untagged_img_path.glob("*.png"))

    mistake_count = 0
    for image_path in tagged_imgs + untagged_imgs:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        tagged = False
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] > threshold_value:
                    tagged = True

        if tagged and "/tagged/" not in str(image_path):
            mistake_count += 1
            print(f"FALSE POSITIVE - {image_path.name}")
        elif not tagged and "/untagged/" not in str(image_path):
            print(f"FALSE NEGATIVE - {image_path.name}")
            mistake_count += 1
    return mistake_count


if __name__ == "__main__":
    main()

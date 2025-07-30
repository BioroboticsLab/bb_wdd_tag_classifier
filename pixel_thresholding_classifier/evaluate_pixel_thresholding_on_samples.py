import csv
from collections import namedtuple
from pathlib import Path

import cv2
from hyperparameters import threshold_value
from inference import TagStatus

SAMPLES_PATH = Path.cwd() / "output" / "samples.csv"

Evaluation = namedtuple("Evaluation", ["total", "accuracy"])


def main():
    """Evaluate performance of pixel thresholding classifier on samples."""
    samples = load_samples_csv(SAMPLES_PATH)
    print(samples)
    for sample in samples:
        label = label_image(Path(sample["sample_path"]))
        print(label)
        sample[f"pixel_threshold_label_at_{threshold_value}"] = label
        dictlist_to_csv(
            samples,
            Path(
                "/home/niklas/Documents/dev/uni/bees/bee-classifier/output/samples.csv"
            ),
        )
    confusion_matrix_cropped_images = evaluate_samples(
        samples, "manual_evaluation_based_on_first_frame"
    )
    confusion_matrix_videos = evaluate_samples(
        samples, "manual_evaluation_based_on_video"
    )
    print(
        f"pixel thresholding with cropped images as ground truth: {confusion_matrix_cropped_images}"
    )
    print(f"pixel thresholding with videos as ground truth: {confusion_matrix_videos}")


def load_samples_csv(path: Path):
    with open(path) as csvfile:
        return list(csv.DictReader(csvfile))


def label_image(image_path: Path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    label = TagStatus.untagged.name
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > threshold_value:
                label = TagStatus.tagged.name
    return label


def dictlist_to_csv(data: list[dict[str, str]], filename: Path):
    with open(filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(data[0].keys())
        writer.writerows([row.values() for row in data])


def evaluate_samples(sample_data, ground_truth_column: str):
    confusion_matrix = dict(
        true_positive=0, false_negative=0, false_positive=0, true_negative=0
    )
    for sample in sample_data:
        label = sample[f"pixel_threshold_label_at_{threshold_value}"]
        if (
            label == TagStatus.tagged.name
            and sample[ground_truth_column] == TagStatus.tagged.name
        ):
            confusion_matrix["true_positive"] += 1
        elif (
            label == TagStatus.untagged.name
            and sample[ground_truth_column] == TagStatus.tagged.name
        ):
            confusion_matrix["false_negative"] += 1
        elif (
            label == TagStatus.tagged.name
            and sample[ground_truth_column] == TagStatus.untagged.name
        ):
            confusion_matrix["false_positive"] += 1
        elif (
            label == TagStatus.untagged.name
            and sample[ground_truth_column] == TagStatus.untagged.name
        ):
            confusion_matrix["true_negative"] += 1
    return confusion_matrix


if __name__ == "__main__":
    main()

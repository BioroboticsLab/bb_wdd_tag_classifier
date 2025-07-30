import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from cnn_classifier.inference import TaggedBeeClassifierConvNet, class_labels


def main():
    parser = init_argparse()
    args: MyArgs = parser.parse_args(namespace=MyArgs())
    cropped_image_dir = Path(args.cropped_image_dir)
    output_dir = Path(args.output_dir)
    classifier = TaggedBeeClassifierConvNet("output/model.pth")
    data = run_classifier_on_all(classifier, cropped_image_dir)
    data = pd.DataFrame.from_dict(data)
    generate_plots_pdfs(data, output_dir)


def generate_plots_pdfs(df: pd.DataFrame, output_dir: Path):
    """
    Creates a multi-page PDF containing a grid of cropped images sorted by tag
    status and the model's confidence.
    """
    df = df.sort_values(by=["class", "confidence"])
    count = df.shape[0]
    plots_per_page = 50
    num_pages = int(np.ceil(count / plots_per_page))
    rows, cols = (10, 5)
    unique_dates = df["date"].unique()
    df_dict: dict[str, pd.DataFrame] = {
        unique_date: pd.DataFrame() for unique_date in unique_dates
    }
    for key in df_dict.keys():
        df_dict[key] = df.loc[df["date"] == key]
        filename = str(output_dir / (str(key) + ".pdf"))
        pdf_pages = PdfPages(filename)
        output_dir.mkdir(parents=True, exist_ok=True)
        day_df = df_dict[key]
        figures = [
            plt.figure(i, figsize=(8.27, 11.69), dpi=100) for i in range(num_pages)
        ]
        last_tagged_idx = 0
        last_untagged_idx = count - 1
        current_idx = None
        for _, row in day_df.iterrows():
            if row["class_label"] == class_labels[0]:
                current_idx = last_tagged_idx
                last_tagged_idx += 1
            else:
                current_idx = last_untagged_idx
                last_untagged_idx -= 1
            current_page = current_idx // plots_per_page
            plt.figure(current_page)
            plt.subplot2grid(
                (rows, cols),
                (
                    (current_idx // cols) % rows,
                    current_idx % cols,
                ),
            )
            with Image.open(str(row.get("cropped_image_path"))) as cropped_image:
                plt.imshow(cropped_image, cmap="gray")
                plt.axis("off")
        for figure in figures:
            figure.savefig(pdf_pages, format="pdf")
        pdf_pages.close()
        plt.close("all")


def run_classifier_on_all(
    classifier: TaggedBeeClassifierConvNet, cropped_image_dir: Path
):
    """
    Applies the classifier to all images in a given directory containing
    cropped images. Uses batches.
    """
    predictions, confidences, paths = classifier.classify_images_from_directory(
        cropped_image_dir, 128
    )
    dates = []
    date_pattern = r"20\d{2}-\d{2}-\d{2}"
    for path in paths:
        date_match = re.search(date_pattern, str(path))
        if date_match is None:
            raise ValueError(f"Failed to extract date info from path: {path}")
        dates.append(date_match.group(0))
    data = {
        "class": predictions,
        "class_label": np.array([class_labels[pred] for pred in predictions]),
        "confidence": confidences,
        "cropped_image_path": paths,
        "date": np.array(dates),
    }
    return data


class MyArgs(argparse.Namespace):
    cropped_image_dir: Path
    output_dir: Path


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Creates a multi-page PDF containing a grid of cropped images sorted by tag status and the model's confidence."
        ),
    )
    parser.add_argument(
        "cropped_image_dir",
        type=Path,
        help="path to directory containing cropped images",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="path to output directory",
    )
    return parser


if __name__ == "__main__":
    main()

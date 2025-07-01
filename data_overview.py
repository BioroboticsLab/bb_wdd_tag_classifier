import argparse
import json
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

from tqdm import tqdm

from daily_data_processing import is_wood_in_frame


def main():
    """
    Gives a daily overview of the number of detections, videos  per dance type
    and how many videos are filtered out by the wood filter.
    """
    parser = init_argparse()
    args: MyArgs = parser.parse_args(namespace=MyArgs())
    zipped_wdd_data_dir = Path(args.zipped_wdd_data_dir)
    data = dict()
    zip_files = sorted(list(zipped_wdd_data_dir.rglob("*")))
    for zip_path in tqdm(zip_files):
        if not zip_path.suffix == ".zip":
            continue
        predicted_class_labels = ["waggle", "activating", "ventilating", "other"]
        day_data = {
            label: {"detections": 0, "videos": 0, "wood filter": 0}
            for label in predicted_class_labels
        }
        with ZipFile(zip_path) as zip_file:
            files = zip_file.namelist()
            metadata_filenames = list(
                filter(lambda filename: filename.endswith(".json"), files)
            )
            for metadata_filename in tqdm(metadata_filenames):
                with zip_file.open(metadata_filename) as metadata_file:
                    json_data = json.load(metadata_file)
                label = json_data["predicted_class_label"]
                day_data[label]["detections"] += 1
                if is_wood_in_frame(json_data):
                    day_data[label]["wood filter"] += 1

                # find matching video file
                video_filename = metadata_filename.replace("waggle.json", "frames.apng")
                if video_filename in files:
                    day_data[label]["videos"] += 1
        date_str = zip_path.stem
        data[date_str] = day_data
    with open("output/data_overview.json", "w") as file:
        json.dump(data, file, indent=2)


class MyArgs(argparse.Namespace):
    zipped_wdd_data_dir: Path
    woodfilter: Optional[bool]


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gives a daily overview of the number of detections and videos per dance type and how many videos would be filtered out by the wood filter"
        ),
    )
    parser.add_argument(
        "zipped_wdd_data_dir",
        type=Path,
        help="path to directory containing zip archives of WDD detection data (APNG video snippets and JSON metadata)",
    )
    return parser


if __name__ == "__main__":
    main()

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

from tqdm import tqdm

from daily_data_processing import is_wood_in_frame, validate_csv_path
from utils.metadata_types import MetadataJson


def main():
    """
    Processes zipped WDD detection data to produce a daily overview of the
    number of detections and video snippets per predicted class label.

    If the --woodfilter flag is set, and a valid --wdd_markers_path is provided,
    also counts how many detections would be filtered out due to wood being
    visible in the frame.
    """
    parser = init_argparse()
    args: MyArgs = parser.parse_args(namespace=MyArgs())
    zipped_wdd_data_dir = Path(args.zipped_wdd_data_dir)
    apply_woodfilter = args.woodfilter
    if apply_woodfilter:
        if args.wdd_markers_path is None:
            parser.error("--woodfilter requires --wdd_markers_path to be set")
        try:
            wdd_markers_path = args.wdd_markers_path
            validate_csv_path(wdd_markers_path)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    data = dict()
    zip_files = sorted(list(zipped_wdd_data_dir.rglob("*")))
    for zip_path in tqdm(zip_files):
        if not zip_path.suffix == ".zip":
            continue
        predicted_class_labels = ["waggle", "activating", "ventilating", "other"]
        day_data = {
            label: {
                "detections": 0,
                "videos": 0,
                **({"wood filter": 0} if apply_woodfilter else {}),
            }
            for label in predicted_class_labels
        }
        with ZipFile(zip_path) as zip_file:
            files = zip_file.namelist()
            metadata_filenames = list(
                filter(lambda filename: filename.endswith(".json"), files)
            )
            for metadata_filename in tqdm(metadata_filenames):
                with zip_file.open(metadata_filename) as metadata_file:
                    json_data: MetadataJson = json.load(metadata_file)
                label = json_data["predicted_class_label"]
                day_data[label]["detections"] += 1
                if apply_woodfilter and is_wood_in_frame(json_data, wdd_markers_path):
                    day_data[label]["wood filter"] += 1

                # find matching video file
                video_filename = metadata_filename.replace("waggle.json", "frames.apng")
                if video_filename in files:
                    day_data[label]["videos"] += 1
        date_str = zip_path.stem
        data[date_str] = day_data
    output_path = Path.cwd() / "output" / "data_overview.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(data, file, indent=2)


class MyArgs(argparse.Namespace):
    zipped_wdd_data_dir: Path
    woodfilter: Optional[bool]
    wdd_markers_path: Optional[Path]


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Processes zipped WDD detection data to produce a daily overview of the number of detections and video snippets per predicted class label."
        ),
    )
    parser.add_argument(
        "zipped_wdd_data_dir",
        type=Path,
        help="path to directory containing zip archives of WDD detection data (APNG video snippets and JSON metadata)",
    )
    parser.add_argument(
        "--woodfilter",
        action="store_true",
        help="enable counting of detections with wood in the frame",
    )
    parser.add_argument(
        "--wdd_markers_path",
        type=Path,
        help="path to CSV file required by the wood filter, contains coordinates of the markers at the corners of the comb",
    )
    return parser


if __name__ == "__main__":
    main()

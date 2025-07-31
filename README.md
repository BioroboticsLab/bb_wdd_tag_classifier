# **bb_wdd_tag_classifier**

This repository provides tools for classifying the tag status (tagged / untagged) of a focal bee inside a WDD video snippet.

## Directory Overview

- `cnn_classifier/`  
  Contains a Convolutional Neural Network (CNN) classifier that determines the tag status based on the center-cropped first frame of a WDD video snippet.

- `pixel_thresholding_classifier/`  
  Contains a rule-based classifier that uses pixel intensity thresholding to detect the presence of a tag.

- `daily_data_processing.py`  
  Runs the CNN-based classifier on a directory of WDD data and processes the video snippets and predicted labels into a format that is compatible with [bb_wdd_label_gui](https://github.com/BioroboticsLab/bb_wdd_label_gui), a browser-based tool for labeling WDD video snippets.  
  The GUI can be used to review multiple snippets at once, correct labels, and add dance type labels.

Other files in the repository support model training, evaluation, data handling, and the thresholding logic.

## Installation

Activate your virtual environment and then install the package:

```bash
conda activate beesbook
pip install git+https://github.com/BioroboticsLab/bb_wdd_tag_classifier.git
```

from enum import Enum
from functools import singledispatch

import numpy as np
from hyperparameters import threshold_value
from PIL import Image


class TagStatus(Enum):
    tagged = 0
    untagged = 1


@singledispatch
def classify_image(image, threshold_value: int = threshold_value) -> TagStatus:
    raise NotImplementedError(f"Unsupported image type: {type(image)}")


@classify_image.register(np.ndarray)
def _(image: np.ndarray, threshold_value: int = threshold_value) -> TagStatus:
    print(image.shape)
    print(threshold_value)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > threshold_value:
                return TagStatus.tagged
    return TagStatus.untagged


@classify_image.register(Image.Image)
def _(cropped_image: Image.Image, threshold_value: int = threshold_value) -> TagStatus:
    for i in range(cropped_image.width):
        for j in range(cropped_image.height):
            pixel = cropped_image.getpixel((i, j))
            if isinstance(pixel, int):
                if pixel > threshold_value:
                    return TagStatus.tagged
            else:
                raise TypeError(f"Expected pixel to be an int: {pixel}")
    return TagStatus.untagged

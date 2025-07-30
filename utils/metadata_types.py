from typing import List, TypedDict


class MetadataJson(TypedDict):
    roi_coordinates: List[List[int]]  # e.g. [[1048, 518], [1298, 768]]
    roi_center: List[int]  # e.g. [1173, 643]
    timestamp_begin: str  # ISO datetime string

    x_coordinates: List[float]
    y_coordinates: List[float]
    responses: List[float]

    frame_timestamps: List[str]  # list of ISO datetime strings
    camera_timestamps: List[str]  # list of ISO datetime strings

    frame_buffer_indices: List[int]
    subsampling: int
    global_roi: List[int]  # e.g. [390, 100, 1280, 736]

    cam_id: str
    waggle_id: int
    predicted_class: int
    predicted_class_label: str
    predicted_class_confidence: float
    waggle_angle: float
    waggle_duration: float
    subdirectory_index: int

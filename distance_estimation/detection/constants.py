KITTI_CLASS_NAMES = [
    "Car",
    "Pedestrian",
    "Van",
    "Cyclist",
    "Truck",
    "Misc",
    "Tram",
    "Person_sitting",
    "DontCare",
]


KITTI_DETECTION_TRAIN_PATH = "data/detection"

KEY_DONT_CARE = "DontCare"

KITTI_COLUMNS = [
    "label",
    "truncated",
    "occluded",
    "alpha",
    "bbox_xmin",
    "bbox_xmax",
    "bbox_ymin",
    "bbox_ymax",
]

KITTI_CLASSNAME_TO_NUMBER = {classname: number for number, classname in enumerate(KITTI_CLASS_NAMES)}

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
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "3d_h",
    "3d_w",
    "3d_l",
    "3d_x",
    "3d_y",
    "distance",
    "rot",
]

KITTI_CLASSNAME_TO_NUMBER = {classname: number for number, classname in enumerate(KITTI_CLASS_NAMES)}

KITTI_DEFAULT_SIZE = (1242, 375)
YOLO_DEFAULT_SIZE = (224, 640)

# vehicle-distance-estimation

TWM Projekt

## KITTI Object Detection with YOLOV8

### Data download

1) Download KITTI Detection Dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d).
2) Unzip necessary files and put to correct directories:

```
. (root dir)
|-data
    |---detection
        |-----testing
        |-------image_2
        |-----training
        |-------image_2
        |-------label_2
```

### Data preprocessing

1) Go to root directory (vehicle-distance-estimation/)

2) Set env:

    ```sh
    export PYTHONPATH=$PWD
    ```

3) Run data preprocessing:

    ```sh
    python distance_estimation/detection/prepare_kitti_data.py
    ```

4) Train model:

    ```sh
    python distance_estimation/detection/train_model.py
    ```

### Results (val)

```
Class               Box(P          R      mAP50  mAP50-95) 
----------------------------------------------------------
all                 0.887      0.825      0.893      0.668
Car                 0.928      0.918      0.964      0.802
Pedestrian          0.873      0.699      0.809      0.466
Van                 0.931      0.909      0.959      0.770
Cyclist             0.843      0.712      0.796      0.554
Truck               0.932      0.964      0.984      0.839
Misc                0.927      0.863      0.914      0.704
Tram                0.917      0.932      0.967      0.779
Person_sitting      0.744      0.607      0.754      0.434
```

Speed: 2.6ms preprocess, 8.7ms inference, 2.5ms postprocess per image at shape (1, 3, 640, 640)
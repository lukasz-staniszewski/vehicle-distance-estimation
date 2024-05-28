# Vehicle Distance Estimation

Team:

1. Łukasz Staniszewski
2. Mateusz Szczepanowski
3. Albert Ściseł

## I. Environment setup

1. Go to root directory (vehicle-distance-estimation/)
2. Use Python 3.11 and install environement with from [requirements](requirements.txt) file.
3. Set PYTHONPATH:

    ```bash
    export PYTHONPATH=$PWD
    ```

## II. Data preprocessing

### Data download

1) Download KITTI Detection Dataset from [here](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d). Especially:
    + Left color images of object data set (12 GB)
    + Camera calibration matrices of object data set (16 MB)
    + Training labels of object data set (5 MB)
2) Unzip necessary files and put to correct directories:

    ```sh
    . (root directory)
    |-data
        |---detection
                |-----testing
                        |-------image_2
                        |-------calib
                |-----training
                        |-------image_2
                        |-------label_2
                        |-------calib
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

## III. KITTI Object Detection with YOLOV8

### Train model

```bash
python distance_estimation/detection/train_model.py
```

### Example training results (on val set)

```sh
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

Speed: 2.6ms preprocess, 8.7ms inference, 2.5ms postprocess per image at shape (1, 3, 640, 640)
```

### Predict per image

```bash
python distance_estimation/detection/predict.py -mp experiments/detection/yolov8-kitti-detection/train/weights/best.pt  -ip data/detection/testing/image_2/000033.png  -op detect_000033.png
```

### Example output image

![yolo_out](https://github.com/lukasz-staniszewski/focus-convolutional-neural-network/assets/59453698/53627712-99a2-454c-aab1-b54108b9d7b8)

## IV. Dummy Distance Prediction (from Bounding-Boxes)

### Create model

```bash
python distance_estimation/dummy_distance_prediction/ddp_prepare.py
```

### Example results

Metric is mean absolute error (in meters).

```bash
Mean Absolute Errors for each class on valid:
Car: 2.1339 m
Pedestrian: 0.9445 m
Van: 5.0367 m
Cyclist: 1.2596 m
Truck: 6.0686 m
Misc: 10.9674 m
Tram: 4.0327 m
Person_sitting: 1.2417 m

Macro Mean Absolute Error on valid: 3.9606 m
Micro Mean Absolute Error on valid: 2.5149 m
```

### Example prediction

```bash
python distance_estimation/dummy_distance_prediction/ddp_predict.py -detmp experiments/detection/yolov8-kitti-detection/train/weights/best.pt -ddpmp distance_estimation/dummy_distance_prediction/model.json  -ip data/detection/testing/image_2/000033.png  -op detect_dist_000033.png
```

### Example image

![ddp_out](https://github.com/lukasz-staniszewski/quantized-depth-estimation/assets/59453698/bc2b806d-0702-477e-88bb-ab2fa8d926fc)

## V. Depth estimation

TODO

## VI. Distance Prediction

TODO

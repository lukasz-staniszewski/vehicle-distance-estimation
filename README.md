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
    export PYTHONPATH=$PWD:$PWD/distance_estimation/depth_prediction/depth_anything:$PWD/distance_estimation/depth_prediction/depth_anything/metric_depth
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

2) Run data preprocessing:

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

![yolo_out](https://github.com/lukasz-staniszewski/focus-convolutional-neural-network/assets/59453698/53627712-99a2-454c-aab1-b54108b9d7b8)

## IV. Dummy Distance Prediction (from Bounding-Boxes)

### Create model

```bash
python distance_estimation/dummy_distance_prediction/ddp_prepare.py
```

### Example results

Metric is mean absolute error (in meters).

```bash
On testing validation...
Mean Absolute Errors for each class:
Car: 2.1666 m
Pedestrian: 1.0545 m
Van: 4.6537 m
Cyclist: 1.1866 m
Truck: 6.4670 m
Misc: 10.6080 m
Tram: 3.6167 m
Person_sitting: 1.4412 m

Macro Mean Absolute Error: 3.8993 m
Micro Mean Absolute Error: 2.4889 m
```

### Example prediction

```bash
python distance_estimation/dummy_distance_prediction/ddp_predict.py -detmp experiments/detection/yolov8-kitti-detection/train/weights/best.pt -ddpmp distance_estimation/dummy_distance_prediction/model.json  -ip data/detection/testing/image_2/000033.png  -op detect_dist_000033.png
```

![dummy distance example](https://github.com/lukasz-staniszewski/quantized-depth-estimation/assets/59453698/cf092a6b-ad4e-40d6-a570-4ab955aa8c78)

## V. Depth estimation

### Setup

Download models to [checkpoints dir](checkpoints/):

```bash
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth

wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt
```

### Relative depth prediction

```bash
python distance_estimation/depth_prediction/predict_depth_relative.py --img-path data/detection/training/image_2/000003.png  --outdir ./
```

### Metric depth prediction

```bash
python distance_estimation/depth_prediction/predict_depth_metric.py --img-in data/detection/training/image_2/000003.png -p local::./checkpoints/depth_anything_metric_depth_outdoor.pt
```

## VI. Distance Prediction (Object detection + depth estimation)

Run sample prediction:

```bash
python distance_estimation/distance_prediction/predict.py -depmn zoedepth -depmp local::./checkpoints/depth_anything_metric_depth_outdoor.pt -detmp experiments/detection/yolov8-kitti-detection/train/weights/best.pt -s center_min -ip data/detection/training/image_2/000003.png -op dist.png
```

![distance example](https://github.com/lukasz-staniszewski/quantized-depth-estimation/assets/59453698/7e380959-6663-48d8-9df6-e33a3c297cd9)

## TODO

1) Split Train/Valid/Test for benchmarking + benchmarking
2) Train ZoeDepth on Kitti for metric depth with Small encoder (faster model)
3) Make depth prediction working with streamlit app

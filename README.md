# Vehicle Distance Estimation

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

3) [Optional] To train Metric Depth-Anything model, download KITTI depth data by using [this script](data/depth/download_kitti_depth.sh). You will end up having such directories:

    ```sh
    . (root directory)
    |-data
        |---depth
                |-----raw
                        |-------2011_09_26
                        |-------2011_09_28
                        |-------...
                |-----masks
                        |-------2011_09_26_drive_0001_sync
                        |-------2011_09_26_drive_0002_sync
                        |-------...
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

### Training results (test set)

```sh
Class               Box(P          R      mAP50  mAP50-95) 
----------------------------------------------------------
all                 0.866      0.792      0.857      0.614
Car                 0.931      0.897      0.955      0.773
Pedestrian          0.863      0.663      0.783      0.449
Van                 0.901      0.852       0.92      0.726
Cyclist             0.867      0.744      0.827      0.535
Truck               0.931      0.921      0.964      0.787
Misc                0.859      0.737      0.825       0.57
Tram                0.863      0.948      0.963      0.709
Person_sitting      0.713      0.572      0.622      0.359

Speed: 0.0ms preprocess, 0.5ms inference, 0.0ms loss, 0.6ms postprocess per image
```

### Predict per image

```bash
python distance_estimation/detection/predict.py -mp experiments/detection/yolov8-kitti-detection/train/weights/best.pt  -ip data/detection/testing/image_2/000033.png  -op detect_000033.png
```

![yolo_out](https://github.com/lukasz-staniszewski/focus-convolutional-neural-network/assets/59453698/53627712-99a2-454c-aab1-b54108b9d7b8)

## IV. Height based distance prediction

Model: YOLOv8 Object Detection + distance prediction based on object height

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
# large
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints_metric_depth/depth_anything_metric_depth_outdoor.pt

# small
wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vits14.pth
TODO
```

### Relative depth prediction

```bash
# large
python distance_estimation/depth_prediction/predict_depth_relative.py --img-path data/detection/training/image_2/000003.png  --outdir ./ --vit-type large
```

### Metric depth prediction

Training:

```bash
python distance_estimation/depth_prediction/depth_anything/metric_depth/train_mono.py -m zoedepth -d kitti --pretrained_resource="" --data_path ./data/depth/raw/ --gt_path ./data/depth/masks/ --gt_path_eval ./data/depth/masks/ --data_path_eval ./data/depth/raw/ --filenames_file ./distance_estimation/depth_prediction/depth_anything/metric_depth/train_test_inputs/kitti_eigen_train_files_with_gt.txt --filenames_file_eval ./distance_estimation/depth_prediction/depth_anything/metric_depth/train_test_inputs/kitti_eigen_test_files_with_gt.txt --workers 8 --vit-encoder-type small --epochs 10
```

Prediction:

```bash
python distance_estimation/depth_prediction/predict_depth_metric.py --vit-type small --img-in data/detection/training/image_2/000003.png --pretrained_resource local::./checkpoints/zoedepth-depthanything-smallvit-10epochs_best.pt
```

## VI. Depth based distance prediction

Model: YOLOv8 Object Detection + Depth-Anything Metric Depth Estimation

Run sample prediction:

```bash
# small
python distance_estimation/distance_prediction/predict.py -depvt small  -depmp local::./checkpoints/zoedepth-depthanything-smallvit-10epochs_best.pt -detmp experiments/detection/yolov8-kitti-detection/train/weights/best.pt -s bbox_median -ip data/detection/training/image_2/000003.png -op dist.png

# large
python distance_estimation/distance_prediction/predict.py -depvt large  -depmp local::./checkpoints/depth_anything_metric_depth_outdoor.pt -detmp experiments/detection/yolov8-kitti-detection/train/weights/best.pt -s bbox_median -ip data/detection/training/image_2/000003.png -op dist.png
```

![distance example](https://github.com/lukasz-staniszewski/quantized-depth-estimation/assets/59453698/7e380959-6663-48d8-9df6-e33a3c297cd9)

## VII. Benchmark results

### General

| Distance prediction method   | Macro MAE | Micro MAE | Large MAE | Medium MAE | Small MAE |
|------------------------------|-----------|-----------|-----------|------------|-----------|
| ***Height based***           | 3.655     | 2.395     | 1.555     | **2.421**  | 4.625     |
| ***Depth based (large)***    |           |           |           |            |           |
| -  centered min              | 5.813     | 4.455     | 2.681     | 5.367      | 5.347     |
| -  centered mean             | 3.683     | 2.588     | 1.641     | 2.940      | 3.666     |
| -  centered median           | 3.578     | 2.550     | 1.577     | 2.925      | 3.600     |
| -  centered percentile       | 4.266     | 3.225     | 1.958     | 3.823      | 4.104     |
| -  bbox min                  | 8.141     | 6.385     | 3.542     | 7.745      | 8.268     |
| -  bbox median               | **3.605** | **2.284** | **1.315** | 2.616      | **3.512** |
| -  bbox mean                 | 4.412     | 2.957     | 1.943     | 3.325      | 4.149     |
| -  bbox percentile           | 4.869     | 3.565     | 2.094     | 4.265      | 4.553     |
| ***Depth based (small)***    |           |           |           |            |           |
| -  centered min              | 6.156     | 4.663     | 2.653     | 5.539      | 6.376     |
| -  centered mean             | 4.121     | 2.908     | 1.640     | 3.235      | 4.988     |
| -  centered median           | 4.098     | 2.903     | 1.593     | 3.254      | 5.002     |
| -  centered percentile       | 4.669     | 3.509     | 1.960     | 4.061      | 5.375     |
| -  bbox min                  | 8.312     | 6.495     | 3.535     | 7.820      | 8.860     |
| -  bbox median               | 4.086     | 2.650     | 1.340     | 2.986      | 4.819     |
| -  bbox mean                 | 4.735     | 3.172     | 1.922     | 3.519      | 5.120     |
| -  bbox percentile           | 5.209     | 3.823     | 2.094     | 4.489      | 5.684     |

### Per class

| Distance prediction method  | Car MAE   | Cyclist MAE | Misc MAE  | Pedestrian MAE  | Person sitting MAE | Tram MAE  | Truck MAE |  Van MAE  |
|-----------------------------|-----------|-------------|-----------|-----------------|--------------------|-----------|-----------|-----------|
| ***Height based***          | 1.953     |   1.163     | 9.350     | **0.923**       | 1.024              | **3.723** | 6.491     | 4.610     |
| ***Depth based (large)***   |           |             |           |                 |                    |           |           |           |
| - centered min              | 4.025     | **1.054**   | 5.100     | 1.270           | 1.266              | 15.214    | 11.033    | 7.544     |
| - centered mean             | 2.280     | 1.710       | 3.131     | 1.178           | 0.818              | 9.701     | 6.650     | 3.993     |
| - centered median           | 2.270     | 1.322       | 3.195     | 1.170           | 0.795              | 9.578     | 6.456     | 3.840     |
| - centered percentile       | 2.908     | 1.108       | 3.712     | 1.170           | 0.938              | 11.202    | 7.817     | 5.273     |
| - bbox min                  | 5.849     | 1.677       | 7.575     | 1.536           | 1.552              | 20.785    | 15.996    | 10.161    |
| - bbox median               | **1.941** | 2.279       | **3.089** | 1.435           | 0.852              | 9.734     | 6.237     | **3.272** |
| - bbox mean                 | 2.553     | 5.867       | 3.226     | 3.123           | 1.447              | 9.538     | **6.136** | 3.410     |
| - bbox percentile           | 3.184     | 1.258       | 4.032     | 1.207           | 1.125              | 13.098    | 9.207     | 5.844     |
| ***Depth based (small)***   |           |             |           |                 |                    |           |           |           |
| - centered min              | 4.201     | 1.514       | 5.470     | 1.476           | 1.224              | 15.796    | 11.975    | 7.590     |
| - centered mean             | 2.557     | 2.424       | 3.383     | 1.443           | 0.626              | 10.474    | 7.812     | 4.257     |
| - centered median           | 2.563     | 2.058       | 3.606     | 1.414           | **0.573**          | 10.600    | 7.739     | 4.234     |
| - centered percentile       | 3.142     | 1.724       | 3.908     | 1.369           | 0.770              | 11.864    | 8.988     | 5.591     |
| - bbox min                  | 5.950     | 1.753       | 7.774     | 1.690           | 1.542              | 21.134    | 16.536    | 10.119    |
| - bbox median               | 2.257     | 3.026       | 3.424     | 1.684           | 0.642              | 10.545    | 7.395     | 3.706     |
| - bbox mean                 | 2.737     | 6.151       | 3.368     | 3.141           | 1.537              | 10.075    | 7.225     | 3.646     |
| - bbox percentile           | 3.406     | 1.760       | 4.310     | 1.335           | 1.033              | 13.476    | 10.244    | 6.104     |

### Size and speed

| Distance prediction method  | Size (MB) | GPU Speed (FPS) | CPU Speed (FPS)  |
|-----------------------------|-----------|-----------------|------------------|
| Height based                | 6         | 88.87           |         TODO     |
| Depth based (small)         | 102       | 25.42           |         TODO     |
| Depth based (large)         | 1306      | 10.30           |         TODO     |

## TODO

1) Make depth prediction working with streamlit app

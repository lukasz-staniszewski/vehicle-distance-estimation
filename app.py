import os
from copy import deepcopy

import streamlit as st
from PIL import Image

from distance_estimation.distance_prediction.helpers import draw_dist_detection_bbox
from distance_estimation.distance_prediction.predict import DistancePredictor

DETECTION_MODEL_PATH = "checkpoints/yolo_best.pt"
DEPTH_MODEL_PATH = "local::./checkpoints/zoedepth-depthanything-smallvit-10epochs_best.pt"
VIT_TYPE = "small"
STRATEGY = "bbox_median"

predictor = DistancePredictor.load(
    vit_type=VIT_TYPE,
    depth_model_path=DEPTH_MODEL_PATH,
    detection_model_path=DETECTION_MODEL_PATH,
    strategy=STRATEGY,
    run_multithreaded=False
)
print("Models loaded...")


def load_image(image_file):
    img = Image.open(image_file)
    return img


def save_image(image, filename):
    image.save(filename)


def process_image(image):
    detections = predictor.predict(image=image)
    print("Detections:", detections)
    img = draw_dist_detection_bbox(image=deepcopy(image), detections=detections)

    output_files = []
    save_image(img, "result.jpg")
    output_files.append("result.jpg")
    return output_files


st.title("Vehicle detection app")
st.write("Upload an image and the model will detect objrcts in the image.")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Detecting...")

    output_files = process_image(image)
    for file in output_files:
        st.image(file, caption="Processed Image with Detected Vehicles.", use_column_width=True)

        with open(file, "rb") as img_file:
            btn = st.download_button(label="Download Image", data=img_file, file_name=file, mime="image/jpeg")

    for file in output_files:
        os.remove(file)

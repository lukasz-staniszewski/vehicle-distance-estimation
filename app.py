import os

import streamlit as st
from PIL import Image
from ultralytics import YOLO

DETECTION_MODEL_PATH = "res/detection/model.pt"

detection_model = YOLO(DETECTION_MODEL_PATH)


def load_image(image_file):
    img = Image.open(image_file)
    return img


def save_image(image, filename):
    image.save(filename)


def process_image(image_path):
    results = detection_model([image_path])
    output_files = []
    for i, result in enumerate(results):
        output_filename = f"result_{i}.jpg"
        result.save(filename=output_filename)
        output_files.append(output_filename)
    return output_files


st.title("Vehicle detection app")
st.write("Upload an image and the model will detect objrcts in the image.")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Detecting...")

    file_path = f"temp_{uploaded_file.name}"
    save_image(image, file_path)

    output_files = process_image(file_path)
    for file in output_files:
        st.image(file, caption="Processed Image with Detected Vehicles.", use_column_width=True)

        with open(file, "rb") as img_file:
            btn = st.download_button(label="Download Image", data=img_file, file_name=file, mime="image/jpeg")

    os.remove(file_path)
    for file in output_files:
        os.remove(file)

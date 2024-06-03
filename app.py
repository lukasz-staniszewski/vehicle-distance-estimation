import os
import tempfile
from copy import deepcopy

import cv2
import numpy as np
import streamlit as st
from moviepy.editor import VideoFileClip
from PIL import Image

from distance_estimation.distance_prediction.helpers import draw_dist_detection_bbox
from distance_estimation.distance_prediction.predict import DistancePredictor
from distance_estimation.dummy_distance_prediction.ddp_predict import DummyDistancePredictor

DETECTION_MODEL_PATH = "checkpoints/yolo_best.pt"
SMALL_DEPTH_MODEL_PATH = "local::./checkpoints/zoedepth-depthanything-smallvit-10epochs_best.pt"
LARGE_DEPTH_MODEL_PATH = "local::./checkpoints/depth_anything_metric_depth_outdoor.pt"
HEIGHT_MODEL_PATH = "distance_estimation/dummy_distance_prediction/model.json"
DEFAULT_STRATEGY = "bbox_median"


@st.cache_resource(show_spinner=False)
def load_predictor(model_size: str) -> DistancePredictor:
    """Load the appropriate predictor model based on the specified size."""
    if model_size == "depth-based-small":
        model = DistancePredictor.load(
            vit_type="small",
            depth_model_path=SMALL_DEPTH_MODEL_PATH,
            detection_model_path=DETECTION_MODEL_PATH,
            strategy=DEFAULT_STRATEGY,
            run_multithreaded=False,
        )
    elif model_size == "depth-based-large":
        model = DistancePredictor.load(
            vit_type="large",
            depth_model_path=LARGE_DEPTH_MODEL_PATH,
            detection_model_path=DETECTION_MODEL_PATH,
            strategy=DEFAULT_STRATEGY,
            run_multithreaded=False,
        )
    elif model_size == "height-based":
        model = DummyDistancePredictor.load(height_model_path=HEIGHT_MODEL_PATH, detection_model_path=DETECTION_MODEL_PATH)
    else:
        raise NotImplementedError("Model size not supported.")
    print("Model loaded...")
    return model


def load_image(image_file: bytes) -> Image.Image:
    """Load an image from the given file."""
    img = Image.open(image_file)
    return img


def save_image(image: Image.Image, filename: str) -> None:
    """Save the image to the specified filename."""
    image.save(filename)


def process_image(image: Image.Image, model_size: str, predictor: DistancePredictor, focal_length: float = None) -> list[str]:
    """Process an image to detect distances and return the list of output files."""
    if model_size == "height-based":
        detections = predictor.predict(image=image, focal_length=focal_length)
    else:
        detections = predictor.predict(image=image)
    print("Detections:", detections)
    img = draw_dist_detection_bbox(image=deepcopy(image), detections=detections)

    output_files = []
    output_file = "result.jpg"
    save_image(img, output_file)
    output_files.append(output_file)
    return output_files


def process_video(video_path: str, model_size: str, predictor: DistancePredictor, focal_length: float = None) -> str:
    """Process a video to detect distances and return the path to the output video."""
    cap = cv2.VideoCapture(video_path)
    output_path = "result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if model_size == "height-based":
            detections = predictor.predict(image=image, focal_length=focal_length)
        else:
            detections = predictor.predict(image=image)
        img = draw_dist_detection_bbox(image=deepcopy(image), detections=detections)
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame)

    cap.release()
    out.release()
    return output_path


def convert_video_for_streamlit(input_path: str) -> str:
    """Convert the video for Streamlit and return the output path."""
    output_path = "web_result.mp4"
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec="libx264")
    return output_path


def main():
    """Main function to run the Streamlit application."""
    model_size = st.sidebar.radio("Choose your model", ("depth-based-small", "depth-based-large", "height-based"))
    predictor = load_predictor(model_size)
    st.session_state["predictor"] = predictor

    st.title("Vehicle distance prediction")
    st.write("Upload an image or video and the model will detect objects and their distance.")

    if model_size == "height-based":
        focal_length = st.sidebar.number_input("Enter camera focal length", min_value=1.0, max_value=1500.0, value=707.0493, step=1.0)
        depth_strategy = None
    else:
        focal_length = None
        depth_strategy = st.sidebar.radio(
            "Choose depth model strategy",
            ("bbox_median", "bbox_min", "bbox_mean", "bbox_percentile", "center_median", "center_min", "center_mean", "center_percentile"),
        )

    if depth_strategy is not None:
        predictor.strategy = depth_strategy

    uploaded_file = st.file_uploader("Choose an image or video...", type=["png", "jpg", "jpeg", "mp4"])

    if uploaded_file is not None:
        if uploaded_file.type in ["image/png", "image/jpeg"]:
            image = load_image(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)
            st.write("Processing...")
            output_files = process_image(image, model_size, predictor, focal_length=focal_length)
            for file in output_files:
                st.image(file, caption="Processed Image with Detected Vehicles.", use_column_width=True)

                with open(file, "rb") as img_file:
                    st.download_button(label="Download Image", data=img_file, file_name=file, mime="image/jpeg")

                os.remove(file)
        elif uploaded_file.type == "video/mp4":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
            st.video(video_path)
            st.write("Processing...")
            output_path = process_video(video_path, model_size, predictor, focal_length=focal_length)
            web_result_path = convert_video_for_streamlit(output_path)

            st.write("Processing complete. You can now download or view the processed video.")
            with open(output_path, "rb") as video_file:
                st.download_button(label="Download Video", data=video_file, file_name="result.mp4", mime="video/mp4")

            st.video(web_result_path)
            os.remove(output_path)
            os.remove(web_result_path)


if __name__ == "__main__":
    main()

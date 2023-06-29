import os
import glob
import torch
from PIL import Image
from io import BytesIO
import streamlit as st
from ultralytics import YOLO
from dotenv import load_dotenv
import pandas as pd


def init():
    """
    Initializes the Streamlit application and sets up the initial layout.

    """
    load_dotenv()
    st.set_page_config(layout="wide")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("logo.jpeg", use_column_width=True)
    with col2:
        st.title("WELD-VISION")
        st.markdown("---")
        st.markdown("A web-app to detect the welds in 2D technical engineering drawings. \
                    The app also generates details such as the count of the welds."
                    )

    st.sidebar.title("Settings")

    st.sidebar.markdown("---")


def set_device():
    """
    Sets the device option for inference (CPU or CUDA).

    Returns:
        str: Selected device option.

    """
    if torch.cuda.is_available():
        device_option = st.sidebar.radio("Select Device", ["cpu", "cuda"], disabled=False, index=0)
    else:
        device_option = st.sidebar.radio("Select Device", ["cpu", "cuda"], disabled=True, index=0)
    return device_option


@st.cache_resource
def load_model(path, device):
    """
    Loads the YOLO model.

    Args:
        path (str): Path to the YOLO model.
        device (str): Device option (cpu or cuda).

    Returns:
        YOLO: Loaded YOLO model.

    """
    model_ = YOLO(path)
    model_.to(device)
    print("model to ", device)
    return model_


def get_custom_classes():
    """
    Allows the user to select custom classes for detection.

    Returns:
        list: List of selected classes.

    """
    model_names = list(model.names.values())
    global assigned_class
    assigned_class = st.sidebar.multiselect("Choose weld types", model_names, default=model_names)
    classes = [model_names.index(name) for name in assigned_class]
    return classes


def infer_image(img, classes, confidence):
    """
    Performs inference on the input image.

    Args:
        img (str): Path to the input image.
        classes (list): List of classes to detect.
        confidence (float): Confidence threshold for detection.

    Returns:
        PIL.Image: Image with predictions drawn.
        int: Total count of detected welds.

    """
    result = model.predict(source=img, conf=confidence, classes=classes)
    image = Image.fromarray(result[0].plot()[:, :, ::-1])
    box_counts = {}
    total_count = 0
    key_count = 0
    for res in result:
        for box in res.boxes:
            # Extract the class label for the bounding box
            class_label = model.names[box.cls.item()]

            # Check if the class label is already in the dictionary
            if class_label in box_counts:
                # Increment the count for the class label
                box_counts[class_label] += 1
            else:
                # Initialize the count for the class label
                box_counts[class_label] = 1

            total_count += 1  # Increment total count

    # Compare the total count with susp_welds and adjust confidence
    if total_count < susp_welds:
        confidence *= total_count / susp_welds
        image, box_counts, confidence = infer_image(img, classes, confidence)
    return image, box_counts, confidence


def download_image(img):
    """
    Downloads the given image.

    Args:
        img (PIL.Image): Image to be downloaded.

    """
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    st.download_button(label="Download", data=byte_im, file_name="Prediction.png", mime="image/jpeg")


def image_input(data_src, classes, confidence):
    """
    Handles image input and displays the input image and prediction.

    Args:
        data_src (str): Data source option (sample data or upload your own data).
        classes (list): List of classes to detect.
        confidence (float): Confidence threshold for detection.

    """
    img_file = None
    if data_src == "Sample data":
        # Get all sample images
        img_path = glob.glob("data/sample_images/*")
        img_slider = st.selectbox("Select a sample image", ("Sample 1", "Sample 2", "Sample 3"))
        img_file = img_path[int(img_slider[-1]) - 1]
    else:
        img_bytes = st.file_uploader("Upload an image", type=["png", "jpeg", "jpg"])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split(".")[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        tab1, tab2 = st.tabs(["Input", "Prediction"])
        with tab1:
            st.image(img_file, use_column_width=True)
        with tab2:
            img, class_count, confidence = infer_image(img_file, classes, confidence)
            st.sidebar.slider("Confidence", min_value=0.1, max_value=1.0, value=confidence)
            download_image(img)
            st.image(img, use_column_width=True)
            df = pd.DataFrame(class_count.items(), columns=['Class', 'Detections'])
            st.table(df)


def pdf_input(data_src):
    """
    Handles PDF document input.

    Args:
        data_src (str): Data source option (sample data or upload your own data).

    """
    pdf_file = st.file_uploader("Upload a document", type=["pdf"])


def process(input_option, data_src, classes, confidence):
    """
    Processes the input based on the selected options.

    Args:
        input_option (str): Selected input type ('image' or 'document').
        data_src (str): Selected input source option ('Sample data' or 'Upload your own data').
        classes (list): List of classes to detect.
        confidence (float): Confidence threshold for detection.

    """
    if input_option == "image":
        image_input(data_src, classes, confidence)
    else:
        pdf_input(data_src)

def conf_score(confidence, total_count, susp_welds, img_file, classes):
    # Compare the total count with susp_welds and adjust confidence
    while total_count < susp_welds:
        confidence *= total_count / susp_welds
        image, class_count, total_count = infer_image(img_file, classes, confidence)
    return image, class_count, confidence
    #print("new conf. score: ", confidence)


def main():
    """
    Main function that sets up the Streamlit application and handles user interactions.

    """
    init()
    global model, MODEL_PATH, total_count
    total_count = 0
    MODEL_PATH = os.environ.get("MODEL_PATH")
    # Upload model
    model_src = st.sidebar.radio("Select a model", ["Weld Detector", "Weld Segmentor"])
    # Device options
    device_option = set_device()
    # Input options
    input_option = st.sidebar.radio("Select input type: ", ["image", "document"], index=0)
    # Input src option
    data_src = st.sidebar.radio("Select input source: ", ["Sample data", "Upload your own data"], index=1)
    # Load model
    if model_src == "Weld Detector":
        model = load_model(MODEL_PATH, device_option)
    else:
        st.markdown(":red[Model unavailable at the moment. Please check back later:)]")
        return ()
    # Custom classes
    classes = get_custom_classes()
    # Suspected weld symbols input from user
    global susp_welds
    susp_welds = st.sidebar.number_input("Suspected Welds", min_value=0, step=1)
    # Confidence slider
    #confidence = st.sidebar.slider("Confidence", min_value=0.1, max_value=1.0, value=0.45)
    confidence = 0.45

    # Process input
    process(input_option, data_src, classes, confidence)
    st.sidebar.markdown("---")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
import os
import glob
import torch
from PIL import Image
from io import BytesIO
import streamlit as st
from ultralytics import YOLO
from dotenv import load_dotenv
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
import pandas as pd

import base64

@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    img = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
                <style>
                [data-testid="stAppViewContainer"] > .main {{
                background-image: url("data:image/png;base64,{img}");
                background-position: center; 
                background-repeat: no-repeat;
                background-size: 100% 100%;
                }}
                
                [data-testid="stHeader"] {{
                background: rgba(0,0,0,0);
                }}
                </style>
                """
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


def login():
    set_png_as_page_bg('utils/loginBG.jpg')
    st.markdown("<h1 style='text-align: center;'>WELD-VISION</h1>",
                unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: center;'>®Weld Detectives</h3>", unsafe_allow_html=True)
    buff, col, buff2 = st.columns([1, 3, 1])
    col.markdown("<hr>", unsafe_allow_html=True)
    col1, col2, col3 , col4, col5, col6, col7, col8, col9 = st.columns(9)
    email = col.text_input("Username/Email")
    password = col.text_input("Password", type="password")
    col.markdown("")
    if col.button("Login", use_container_width=True):
        if email and password:
            # Perform login authentication here
            # If login is successful, set the logged_in  variable
            logged_in  = True
            # Update session state
            st.session_state.logged_in  = logged_in 
    col.markdown("Not a member? Sign up!&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;Forgot Password?", unsafe_allow_html=True)


def init():
    """
    Initializes the Streamlit application and sets up the initial layout.

    """
    # set_png_as_page_bg('6663961.jpg')
    load_dotenv()
    global assigned_class
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("utils/logo.jpeg", use_column_width=True)
    with col2:
        st.title("WELD-VISION")
        st.markdown("---")
        st.markdown("A web-app to detect the welds in 2D technical engineering drawings. \
                    The app also generates details such as the count of the welds."
        )
    if st.sidebar.button("Logout"):
        logged_in  = False
        # Update session state
        st.session_state.logged_in  = False
    page = st.sidebar.selectbox("", ("Home", "About", "Contact"))
        
    return page


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


def infer_image(img, classes, confidence = 0.45):
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
    result = model.predict(source=img, conf=confidence, classes=classes, save_txt=None)
    save_file_name1 = img.split('/')[-1]
    # print(save_file_name1)
    save_file_name = save_file_name1.split('.')[0]
    # print(save_file_name)
    with open(f"predictions\{save_file_name}.txt", 'w') as file:
      for idx, prediction in enumerate(result[0].boxes.xywhn): 
          cls = int(result[0].boxes.cls[idx].item())
          # Write line to file in YOLO label format : cls x y w h
          file.write(f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")
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
    if ((total_count < susp_welds) and not (confidence < 0.05)): 
        confidence -= 0.05
        print("Recalculating confidence: ", round(confidence, 2))
        image, box_counts, confidence = infer_image(img, classes, confidence)
    # if ((total_count < susp_welds) and not (confidence < 0.05)):
    #     confidence *= susp_welds / total_count
    #     image, box_counts, confidence = infer_image(img, classes, confidence)
    return image, box_counts, confidence


    return image, box_counts


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

def save_corrected_image(source_path):
    import shutil
    try:
        if "sample" in source_path:
            shutil.copy(source_path, "retrain/images/sample_data")
        else:
            shutil.copy(source_path, "retrain/images/")
        print("File copied successfully.")
    except IOError as e:
        print(f"Unable to copy file. {e}")
    except:
        print("An error occurred while copying the file.")
    
def relabel(img_file, classes):
    # st.set_option("deprecation.showfileUploaderEncoding", False)
    idm = ImageDirManager("img_dir")

    st.session_state["files"] = idm.get_all_files()
    st.session_state["annotation_files"] = idm.get_exist_annotation_files()
    st.session_state["image_index"] = 0
    # else:
    idm.set_all_files(st.session_state["files"])
    idm.set_annotation_files(st.session_state["annotation_files"])
    img_file_name = img_file
    img_path = os.path.join("", img_file_name)
    im = ImageManager(img_path)
    img = im.get_img()
    resized_img = im.resizing_img(max_height=1420, max_width=1420)
    resized_rects = im.get_resized_rects()
    rects = st_img_label(resized_img, box_color="red", rects=resized_rects)

    def annotate():
        im.save_annotation()
        save_corrected_image(img_file)
        image_annotate_file_name = img_file_name.split(".")[0] + ".txt"
        if image_annotate_file_name not in st.session_state["annotation_files"]:
            st.session_state["annotation_files"].append(image_annotate_file_name)
        # next_annotate_file()

    if rects:
        st.button(label="Save", on_click=annotate)
        preview_imgs = im.init_annotation(rects)

        for i, prev_img in enumerate(preview_imgs):
            prev_img[0].thumbnail((200, 200))
            col1, col2 = st.columns(2)
            with col1:
                col1.image(prev_img[0])
            with col2:
                default_index = 0
                if prev_img[1]:
                    default_index = classes.index(prev_img[1])

                select_label = col2.selectbox(
                    "Label", classes, key=f"label_{i}", index=default_index
                )
                im.set_annotation(i, classes.index(select_label))


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
        img_path = glob.glob("data/sample_data/*")
        img_slider = st.selectbox("Select a sample image", ("Sample 1", "Sample 2", "Sample 3"))
        img_file = img_path[int(img_slider[-1]) - 1]
    else:
        img_bytes = st.file_uploader("Upload an image", type=["png", "jpeg", "jpg"])
        if img_bytes:
            img_name = img_bytes.name
            img_file = "data/uploaded_data/" + img_name
            Image.open(img_bytes).save(img_file)
            

    if img_file:
        tab1, tab2, tab3 = st.tabs(tabs)
        with tab1:
            st.image(img_file, use_column_width=True)
        with tab2:
            img, class_count, confidence = infer_image(img_file, classes, confidence)
            st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=confidence)
            download_image(img)
            # st.button(label="relabel", on_click=relabel(img_file, classes))
            st.image(img, use_column_width=True)
            df = pd.DataFrame(class_count.items(), columns=['Class', 'Detections'])
            st.table(df)
        with tab3:
            relabel(img_file, classes)
    return img_file


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
        img_file = image_input(data_src, classes, confidence)

    else:
        pdf_input(data_src)
    return img_file


def format_time(seconds):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def retrain():
    # Text file input
    annotation_file = st.text_input("Enter annotation path", value="./retrain/corrections/labels")

    # Image path input
    image_path = st.text_input("Enter image path", value="./retrain/corrections/images")

    # Model selection dropdown
    model_selection = st.selectbox("Choose a Model", ("Detection-1","Detection-2"))
    import time
    # Checkbox to vary hyperparameters
    vary_hyperparameters = st.checkbox("Advanced Options")

    if vary_hyperparameters:
        col1, col2, col3 = st.columns(3)
        # Radio button for a binary choice
        choice = col1.number_input("Number of Epochs", value=100, step = 100, max_value=1000)
        
        image_path = col1.text_input("Set output path", placeholder="Where do you want to save the model?")

        image_path = col1.markdown("Pause training at")
 
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        # Number input for hours
        with col1:
            hours = st.number_input("Hours", min_value=0, max_value=12, value=0, step=1)

        # Number input for minutes
        with col2:
            minutes = st.number_input("Minutes", min_value=0, max_value=59, value=0, step=1)

        # Dropdown for AM/PM selection
        with col3:
            am_pm = st.selectbox("", ("AM", "PM"))

    success_text = st.empty()
    if st.button("Begin Training"):
        start_time = time.time()
        elapsed_time = 0
        progress_bar = st.progress(0)
        progress_text = st.empty()
        success_text.empty()
        with st.spinner("Training in progress..."):
            warning_text = st.warning("Do not refresh or close the app!")
            while elapsed_time < 10:  # Change the condition based on your stopping criteria
                progress = round(elapsed_time / 10, 1)  # Normalize progress value
                progress_bar.progress(progress)
                formatted_time = format_time(elapsed_time)
                progress_text.text(f"Progress: {progress * 100}% | Elapsed Time: {formatted_time}")
                elapsed_time = round(time.time() - start_time)
                time.sleep(1)
                if progress == 1.0:
                    break

        success_text = st.success(f"Training completed! Model saved at '{'god/knows/where/'}'")
        warning_text.empty()
        
        
def contact():
    st.title("Contact Form")
    # Contact form inputs
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    attachment = st.file_uploader("Attach a photo/file")
    # Submit button
    if st.button("Submit"):
        if name and email and message:
            # send_email(name, email, message)
            st.success("Message sent successfully!")
        else:
            st.error("Please fill in all the fields.")
        
                    
def main():
    """
    Main function that sets up the Streamlit application and handles user interactions.

    """
    page = init()
    if page == "Home":
        st.sidebar.title("Settings")
        st.sidebar.markdown("---")
        global model, MODEL_PATH, total_count, tabs
        total_count = 0 
        MODEL_PATH = os.environ.get("MODEL_PATH")
        tabs = ["Input", "Prediction", "Relabel"]

        # Upload model
        mode = st.sidebar.selectbox("Choose a Mode", ("Weld Detection", "Weld Segmentation", "Retrain"))
        # Device options
        device_option = set_device()
        # Load model
        if mode == "Weld Detection":
            model = load_model(MODEL_PATH, device_option)
        elif mode == "Weld Segmentation":
            st.warning("Model unavailable at the moment. Please check back later:)")
            return ()
        elif mode == "Retrain":
            retrain()
            return()
            # Input options
        input_option = st.sidebar.radio("Select input type: ", ["image", "document"], index=0)
        # Input src option
        data_src = st.sidebar.radio("Select input source: ", ["Sample data", "Upload your own data"], index=1)
        # Custom classes
        classes = get_custom_classes()
        global susp_welds
        if 'expected_count' not in st.session_state:
            st.session_state.expected_count = 0
        #suspected weld symbols input from user
        susp_welds = st.sidebar.number_input("Expected Count", min_value=0, step=1, value=st.session_state.expected_count)
        confidence = 0.45
        # Process input
        process(input_option, data_src, classes, confidence)

        st.sidebar.markdown("---")
    elif page == "About":
        st.success("Thanks for showing interest in us!")
    elif page == "Contact":
        contact()


if __name__ == "__main__":
    try:
        # Check the user's login state
        if 'logged_in' not in st.session_state or not st.session_state.logged_in:       
            st.set_page_config(layout="centered")     
            login()
        else:
            st.set_page_config(layout="wide")
            main()
    except SystemExit:
        pass
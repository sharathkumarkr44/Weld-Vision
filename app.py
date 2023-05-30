import glob
import streamlit as st
from PIL import Image
import torch
from ultralytics import YOLO

st.set_page_config(layout="wide")

cfg_model_path = 'models/yolov8.pt'
model = None
confidence = .25

def image_input(data_src, classes):
    img_file = None
    if data_src == 'Sample data':
        # get all sample images
        img_path = glob.glob('data/sample_images/*')
        img_slider = st.slider("Select a sample image", min_value=1, max_value=len(img_path), step=1)
        img_file = img_path[img_slider - 1]
    else:
        img_bytes = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "data/uploaded_data/upload." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        tab1, tab2 = st.tabs(["Input", "Prediction"])
        with tab1:
            st.image(img_file, caption="Selected Image", use_column_width=True)
        with tab2:
            img = infer_image(img_file, classes)
            st.image(img, caption="Model prediction", use_column_width=True)
            from io import BytesIO
            buf = BytesIO()
            img.save(buf, format="JPEG")
            byte_im = buf.getvalue()
            col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
            with col5:
                btn = st.download_button(
                        label="Download",
                        data=byte_im,
                        file_name="Prediction.png",
                        mime="image/jpeg",
                        )

def pdf_input(data_src):
    vid_bytes = st.file_uploader("Upload a document", type=['pdf'])

def infer_image(img, classes):
    result = model.predict(source=img, conf=confidence, classes = classes)
    image = Image.fromarray(result[0].plot()[:,:,::-1])
    return image

@st.cache_resource 
def load_model(path, device):
    model_ = YOLO(path)
    model_.to(device)
    print("model to ", device)
    return model_

def main():
    # global variables
    global model, confidence, cfg_model_path
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("logo.jpeg", use_column_width=True)
    with col2:
        st.title("WELD-VISION")
        st.markdown("---")
        st.markdown("A web-app to detect the welds in 2D technical engineering drawings. The app also generates details such as the count of the welds.")
        
    # Display the description
    st.sidebar.title("Settings")
    
    st.sidebar.markdown("---")

    # upload model
    model_src = st.sidebar.radio("Select a model", ["Weld Detector", "Weld Segmentor"])

    # device options
    if torch.cuda.is_available():
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
    else:
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

    # input options
    input_option = st.sidebar.radio("Select input type: ", ['image', 'document'], index=0)

    # input src option
    data_src = st.sidebar.radio("Select input source: ", ['Sample data', 'Upload your own data'], index=1)
    
    # load model
    model = load_model(cfg_model_path, device_option)

    # custom classes
    model_names = list(model.names.values())
    assigned_class = st.sidebar.multiselect("Choose weld types", model_names, default=model_names)
    classes = [model_names.index(name) for name in assigned_class]

    # confidence slider
    confidence = st.sidebar.slider('Confidence', min_value=0.1, max_value=1.0, value=.45)

    if input_option == 'image':
        image_input(data_src, classes)
    else:
        pdf_input(data_src)
    
    st.sidebar.markdown("---")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass

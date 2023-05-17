import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# Set page title
st.set_page_config(page_title="Weld-Vision")

# Load and display the logo or image
logo = Image.open("logo.jpg")
st.image(logo, width=200)

# Define the layout of the page
st.title("Weld Detection Web App")
st.write("Upload one or more images and select the types of welds to detect.")

# Create a file uploader widget for uploading images
uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# setting a boolean flag to track if any file has been uploaded
files_uploaded = False

# Define the options for the weld type selection
weld_types = ["Fillet", "Bead", "Groove", "L-Joint", "T-joint", "Corner joint", "Lap joint", "Edge joint"]

# Create a multiselect sidebar for selecting the types of welds to detect
selected_weld_types = st.sidebar.multiselect("Select Weld Types", weld_types, default = weld_types)

# Display the uploaded image
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        files_uploaded = True

# Create a button for detecting objects in the uploaded image (only if files have been uploaded)
if files_uploaded and st.button("Detect"):
    selected_images = []
    # Perform object detection based on the selected weld types
    if "Fillet" in selected_weld_types:
        st.write("Performing object detection for Fillet welds.")
    if "Bead" in selected_weld_types:
        st.write("Performing object detection for Bead welds.")
    if "Groove" in selected_weld_types:
        st.write("Performing object detection for Groove welds.")
    if "L-Joint" in selected_weld_types:
        st.write("Performing object detection for L-Joint welds.")
    if "T-joint" in selected_weld_types:
        st.write("Performing object detection for T-joint welds.")
    if "Corner joint" in selected_weld_types:
        st.write("Performing object detection for Corner joint welds.")
    if "Lap joint" in selected_weld_types:
        st.write("Performing object detection for Lap joint welds.")
    if "Edge joint" in selected_weld_types:
        st.write("Performing object detection for Edge joint welds.")
    for uploaded_file in uploaded_files:
        processed_image = Image.open(uploaded_file) # Placeholder for the processed image
        st.image(processed_image, caption="Processed Image", use_column_width=True)
        selected_images.append(processed_image)
    
    #  Checkbox or multiselect for selecting images to download
    selected_images_to_download = st.multiselect("Select Images to Download", range(len(selected_images)),
                                                 format_func=lambda i:f"Processed Image {i+1}")

    if st.button("Download"):
        # Prepare the selected images for download
        for i in selected_images_to_download:
            image = selected_images[i]
            # Convert image to bytes
            image_bytes = BytesIO()
            image.save(image_bytes, format='PNG')
            # Download link
            href = f'<a href="data:application/octet-stream;base64,{base64.b64encode(image_bytes.getvalue()).decode()}" download="processed_image{i+1}.png">Download Processed Image {i+1}</a>'
            st.markdown(href, unsafe_allow_html=True)
            #newline

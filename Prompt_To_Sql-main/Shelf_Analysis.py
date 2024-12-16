import streamlit as st
import torch
from PIL import Image
import os
import requests

MODEL_URL = 'https://your-cloud-storage-link/best.pt'  # Replace with your hosted link
MODEL_PATH = 'models/best.pt'

os.makedirs('models', exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print(f"Downloading model from {MODEL_URL}...")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):  # Download in chunks
            f.write(chunk)
    print("Model downloaded successfully!")

# Load your vehicle detection model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)  # Adjust the path to your model

# Ensure folders exist
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

st.title("Shelf Stock Analysis")
st.write("Upload an image to detect empty shelves using YOLOv5")

uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])


def process_image(file_path):
    # Load and detect vehicles in the image
    image = Image.open(file_path)
    results = model(image)  # Run detection

    # Render and save processed image
    processed_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(file_path))
    rendered_image = results.render()[0]
    Image.fromarray(rendered_image).save(processed_image_path)
    return processed_image_path, image.size  # Return original dimensions


if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())

    # Process the uploaded image
    st.write("Processing Image...")
    processed_file_path, original_size = process_image(file_path)
    st.write(f"Original Image Dimensions: {original_size[0]}x{original_size[1]}")

    # Display the processed image with original width and height
    st.image(processed_file_path, caption='Processed Image', width=original_size[0])

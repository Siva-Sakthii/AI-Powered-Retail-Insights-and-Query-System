import streamlit as st
import torch
from PIL import Image
import cv2
import os

# Load your vehicle detection model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best.pt')  # Adjust the path to your model

# Ensure folders exist
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

st.title("Vehicle Detection System")
st.write("Upload an image or video to detect vehicles using YOLOv5")

uploaded_file = st.file_uploader("Choose an image or video file", type=['png', 'jpg', 'jpeg', 'mp4', 'avi'])

def process_image(file_path):
    # Load and detect vehicles in the image
    image = Image.open(file_path)
    results = model(image)  # Run detection

    # Render and save processed image
    processed_image_path = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(file_path))
    rendered_image = results.render()[0]
    Image.fromarray(rendered_image).save(processed_image_path)
    return processed_image_path

def process_video(file_path):
    # Open and process video
    cap = cv2.VideoCapture(file_path)
    output_file = os.path.join(PROCESSED_FOLDER, 'processed_' + os.path.basename(file_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on each frame
        results = model(frame)
        processed_frame = results.render()[0]  # Get processed frame
        out.write(processed_frame)  # Write to output video

    cap.release()
    out.release()
    return output_file

if uploaded_file is not None:
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.read())

    if uploaded_file.name.lower().endswith(('png', 'jpg', 'jpeg')):
        st.write("Processing Image...")
        processed_file_path = process_image(file_path)
        st.image(processed_file_path, caption='Processed Image', use_column_width=True)

    elif uploaded_file.name.lower().endswith(('mp4', 'avi')):
        st.write("Processing Video...")
        processed_file_path = process_video(file_path)
        st.video(processed_file_path)

    else:
        st.error("Unsupported file format. Please upload an image or video.")

import streamlit as st
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import io
import matplotlib.pyplot as plt
import time
import cv2
from pathlib import Path

# Page configuration
st.set_page_config(page_title="Object Detection for Self-Driving Cars", layout="wide")

# Title and description
st.title("Object Detection for Self-Driving Cars")
st.markdown("""
This app allows you to detect objects in images and videos relevant to autonomous driving scenarios.
Upload your content and see real-time detection of vehicles, pedestrians, and other road elements.
""")

# Classes
classes = ['Car', 'Pedestrian', 'Van', 'Cyclist', 'Truck', 'Misc', 'Tram', 'Person_sitting']

# Model loading state management
@st.cache_resource
def load_model():
    # Load the model (assuming the model is in the same directory as the app)
    model = YOLO("best.pt")
    return model

# Load the model
with st.spinner("Loading object detection model..."):
    model = load_model()
    st.success("Model loaded successfully!")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["Image Upload", "Video Upload", "Batch Processing", "About"])

with tab1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="single_image")
    
    col1, col2 = st.columns(2)
    
    # Confidence threshold
    conf_threshold = col1.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05, key="conf_img")
    
    # IOU threshold
    iou_threshold = col2.slider("IOU Threshold", min_value=0.1, max_value=1.0, value=0.45, step=0.05, key="iou_img")
    
    if uploaded_file is not None:
        # Create a temporary file to store the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            img_path = tmp_file.name
        
        col1, col2 = st.columns(2)
        
        # Display the original image
        img = Image.open(uploaded_file)
        col1.header("Original Image")
        col1.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Make predictions
        start_time = time.time()
        results = model.predict(img_path, conf=conf_threshold, iou=iou_threshold)
        inference_time = time.time() - start_time
        
        # Display results
        col2.header("Detection Result")
        
        # Get the result image with bounding boxes
        result_img = Image.fromarray(results[0].plot())
        col2.image(result_img, caption="Detected Objects", use_column_width=True)
        
        # Display detection details
        st.subheader("Detection Details")
        st.write(f"Inference Time: {inference_time:.3f} seconds")
        
        # Display detection metrics
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # Create a DataFrame with the results
            data = []
            for i, box in enumerate(boxes):
                class_id = int(box.cls.item())
                class_name = classes[class_id]
                confidence = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                data.append({
                    "Object": i+1,
                    "Class": class_name,
                    "Confidence": f"{confidence:.3f}",
                    "Bounding Box": f"[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
                })
            
            import pandas as pd
            df = pd.DataFrame(data)
            st.table(df)
        else:
            st.info("No objects detected in the image.")
        
        # Clean up the temporary file
        os.unlink(img_path)

with tab2:
    st.header("Upload Video")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"], key="video_upload")
    
    # Add video configuration options
    st.subheader("Video Processing Options")
    col1, col2, col3 = st.columns(3)
    
    # Confidence threshold for video
    conf_threshold_video = col1.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.3, step=0.05, key="conf_video")
    
    # Frame processing rate (skip frames)
    frame_skip = col2.slider("Process every N frames", min_value=1, max_value=10, value=3, step=1)
    
    # Resolution scaling
    resolution_scale = col3.slider("Resolution Scale", min_value=0.25, max_value=1.0, value=0.5, step=0.25)
    
    if uploaded_video is not None:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_video.getvalue())
            video_path = tmp_file.name
        
        # Process the video
        if st.button("Process Video"):
            st.video(uploaded_video)
            
            # Video processing logic
            output_path = "processed_video.mp4"
            
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resolution_scale)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resolution_scale)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process frames
            st.text("Processing video... Please wait.")
            progress_bar = st.progress(0)
            
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every Nth frame
                if frame_count % frame_skip == 0:
                    # Resize frame
                    frame = cv2.resize(frame, (width, height))
                    
                    # Save frame to temp file for processing
                    temp_frame_path = f"temp_frame_{frame_count}.jpg"
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # Process with YOLO
                    results = model.predict(temp_frame_path, conf=conf_threshold_video)
                    
                    # Convert to OpenCV format for saving
                    result_frame = results[0].plot()
                    result_frame = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
                    
                    # Write processed frame
                    out.write(result_frame)
                    
                    # Remove temp frame
                    os.remove(temp_frame_path)
                else:
                    # Write original frame if skipping
                    resized_frame = cv2.resize(frame, (width, height))
                    out.write(resized_frame)
                
                # Update progress
                frame_count += 1
                progress_bar.progress(min(frame_count / total_frames, 1.0))
            
            # Release resources
            cap.release()
            out.release()
            
            # Display processed video
            st.success("Video processing complete!")
            st.video(output_path)
            
            # Option to download the processed video
            with open(output_path, "rb") as file:
                btn = st.download_button(
                    label="Download processed video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )
            
            # Clean up
            os.remove(output_path)
        
        # Clean up the temporary file
        os.unlink(video_path)

with tab3:
    st.header("Batch Processing")
    
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_images")
    
    if uploaded_files:
        batch_conf = st.slider("Batch Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
        
        if st.button("Process All Images"):
            st.write(f"Processing {len(uploaded_files)} images...")
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Process each image
            for i, file in enumerate(uploaded_files):
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file.write(file.getvalue())
                    img_path = tmp_file.name
                
                # Make predictions
                results = model.predict(img_path, conf=batch_conf)
                
                # Get the result image with bounding boxes
                result_img = Image.fromarray(results[0].plot())
                
                # Display original and result side by side
                col1, col2 = st.columns(2)
                col1.image(Image.open(file), caption=f"Original: {file.name}", use_column_width=True)
                col2.image(result_img, caption="Detection Result", use_column_width=True)
                
                # Clean up temporary file
                os.unlink(img_path)
                
                # Add a separator between images
                st.markdown("---")
            
            # Clear progress bar when done
            progress_bar.empty()
            st.success("All images processed!")

with tab4:
    st.header("About This Application")
    
    # Display general information
    st.markdown("""
    ### Object Detection for Self-Driving Cars
    
    This application uses computer vision to detect and classify objects relevant to autonomous driving scenarios.
    
    #### Detectable Objects:
    - Cars
    - Pedestrians
    - Vans
    - Cyclists
    - Trucks
    - Misc objects
    - Trams
    - People sitting
    
    #### Use Cases:
    - Safety testing for autonomous vehicles
    - Analysis of traffic scenarios
    - Understanding urban road environments
    - Educational purposes for self-driving car technology
    """)
    
    # Visualize class distribution
    st.subheader("Object Classes")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, [1] * len(classes), color='skyblue')
    ax.set_ylabel('Objects')
    ax.set_title('Detectable Objects in Driving Scenarios')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

# Add a sidebar with additional options
with st.sidebar:
    st.header("Detection Settings")
    
    # Detection mode
    detection_mode = st.radio(
        "Detection Mode",
        ("Standard", "High Accuracy", "Fast Detection")
    )
    
    if detection_mode == "High Accuracy":
        st.info("High Accuracy mode uses stricter thresholds but may be slower.")
    elif detection_mode == "Fast Detection":
        st.info("Fast Detection mode prioritizes speed with somewhat lower accuracy.")
    
    # Color scheme
    color_scheme = st.selectbox(
        "Visualization Color Scheme",
        ("Standard", "High Contrast", "Pastel")
    )
    
    st.header("Instructions")
    st.markdown("""
    1. Select the appropriate tab for your content (image, video, or batch)
    2. Upload your files
    3. Adjust detection parameters if needed
    4. Click the process button to see results
    """)
    
    # Add an example image
    st.header("Sample Detection")
    st.image("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg", 
             caption="Example Detection", use_column_width=True)
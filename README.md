# Deepfake Detection Suite

This repository contains tools for detecting deepfakes in **videos** and **images** using state-of-the-art machine learning models. It supports batch processing, configurable thresholds, and high-accuracy predictions.

---

## Available Models and Functionalities

### **Video Deepfake Detection**
1. **EfficientNet Variants**:
   - **EfficientNetB4**: General-purpose deepfake detection.
   - **EfficientNetB4ST**: Enhanced variant with self-attention for improved accuracy.
   - **EfficientNetAutoAttB4**: Focuses on automated attention mechanisms.
   - **EfficientNetAutoAttB4ST**: Best-performing model with self-attention and high precision.
   
   **Key Features**:
   - Frame-by-frame face detection (BlazeFace).
   - Batch processing of videos with CSV output.
   - Classification: **Real**, **Fake**, or **Uncertain**.

   **Setup**:
   - **README Location**: `efficientNet/app_info.md`

    Following is a screenshot of it running on the RescueBox app. 

    ![efficient_model](efficientNet/server.png)

2. **XceptionNet**:
    - **XceptionNet**: Alternative model for video deepfake detection.
    
    **Key Features**:
    - Frame-by-frame face detection (MTCNN).
    - Supports video processing with CSV output.
    - Classification: **Real**, **Fake**.
    
    **Setup**:
    - **README Location**: `video_detector/app_info.md`

    Following is a screenshot of it running on the RescueBox app. 

    ![xception_model](video_detector/server.png)

---

### **Image Deepfake Detection**

1. **BNext-Based Binary Classifier**:
   - Backbone: Pre-trained neural network (e.g., CelebA, DFFD datasets).
   - Supports face cropping with RetinaFace for focused detection.
   - Predictions: **Real**, **Fake**, **Uncertain** (confidence scores provided).

   **Key Features**:
   - Batch image classification via Flask server (`model_server.py`).
   - RescueBox GUI for easy interaction and output CSV generation.


Given a directory of images, the model can detect whether or not a image has been altered through AI tech (deepfake, faceswap etc). It can also extract face regions from the image and then check for any fakery. It can use the RescueBox app to give a nice UI for easy interaction. Following is a screenshot of it running on the RescueBox app. More info, including evaluation metrics, can be found at `image_model/img-app-info.md`. 

![img_model](image_model/image_server.png)

---

## Setup

For detailed setup and usage instructions, refer to the full documentation.


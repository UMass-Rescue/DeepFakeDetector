# Video Deepfake Detection Application

## Overview
This application provides a robust solution for detecting deepfake videos using advanced machine learning techniques. By leveraging state-of-the-art neural network architectures, the tool can analyze video files and classify them as real, fake, or uncertain with high accuracy.

## Key Features
- Multi-model deepfake detection support
- Batch video processing
- Configurable detection thresholds
- CSV output for easy result analysis
- Supports multiple video file formats

## Supported Models
1. EfficientNetB4
2. EfficientNetB4ST
3. EfficientNetAutoAttB4
4. EfficientNetAutoAttB4ST

## System Requirements
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Sufficient storage for video processing

## Project Structure
```
efficientNet/
│
├── server.py               # Main Flask server implementation
├── detect_deepfakes.py      # Core video processing and deepfake detection logic
├── evalpipeline.py          # Command-line video processing script for evluation of models
├── requirements.txt        # Python package dependencies
├── architectures/               # Neural network model definitions        
├── isplutils/                # Data transformation utils
├── blazeface/             # Face detection model
└── app_info.md             # Application documentation
```

## Key Components
1. **Deepfake Detector (`detect_deepfakes.py`)**: 
   - Implements a sophisticated neural network-based deepfake detection mechanism
   - Utilizes advanced machine learning architectures like EfficientNet for video analysis
   - Performs face extraction and probabilistic prediction using the following core functionalities:
     - Face detection with BlazeFace model
     - Frame-level feature extraction
     - Probability computation using sigmoid activation
     - Classification of videos as REAL, FAKE, or UNCERTAIN based on configurable probability thresholds
   - Supports multiple pre-trained model variants for flexible detection
   - Processes videos by analyzing individual frames and aggregating predictions
   - Provides robust error handling and device-agnostic computation (CPU/GPU)

2. **Evaluation Pipeline (`evalpipeline.py`)**: 
   - Serves as a comprehensive command-line interface for large-scale video analysis
   - Automates the process of discovering and processing video files across entire directory structures
   - Provides key features including:
     - Recursive video file discovery with support for multiple file extensions
     - Batch processing of videos using different machine learning models
     - Automatic result aggregation and statistical summarization
   - Enables systematic evaluation by:
     - Scanning specified directories for video content
     - Applying multiple deepfake detection models
     - Generating detailed CSV reports for each model
     - Producing summary statistics of detection results
   - Offers flexible configuration through interactive input
   - Supports comprehensive model comparison by processing the same dataset across different neural network architectures

3. **MLServer (`server.py`)**: 
   - Creates a Flask-based server for processing video files
   - Provides a flexible API for deepfake detection
   - Supports batch processing of multiple videos
   - Offers comprehensive configuration options including:
     - Model selection
     - Real and fake probability thresholds
     - Output directory specification
   - Generates standardized CSV output for easy result interpretation

## Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/aravadikesh/DeepFakeDetector.git
cd efficientNet
```

### 2. Create Virtual Environment
```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# OR using conda
conda create -n deepfake-detector python=3.11
conda activate deepfake-detector
```

### 3. Install Required System Dependencies

```bash
pip install efficientnet-pytorch
pip install -U git+https://github.com/albu/albumentations 
pip install -r requirements.txt
```

## Usage Methods

### Command-Line Processing
Run `python evalpipeline.py` and:
- Enter the directory path containing videos
- The script will automatically:
  - Detect video files
  - Process each video through multiple models
  - Generate CSV result files

### Web Server Mode
Run `python server.py` to start a Flask-based web interface for:
- Batch video processing
- Interactive threshold configuration
- Graphical result export

## Detection Workflow
1. Face extraction from video frames
2. Neural network probability prediction
3. Classification based on configurable thresholds:
   - ≤ 0.2: Classified as REAL
   - 0.2 - 0.75: Classified as UNCERTAIN
   - ≥ 0.75: Classified as FAKE

## Output
Generates a CSV file with the following columns:
- `video_path`: Source video filename
- `deepfake_probability`: Confidence score
- `prediction`: REAL/FAKE/UNCERTAIN status

## Customization
Modify detection thresholds in configuration:
- `real_threshold`: Default 0.2
- `fake_threshold`: Default 0.8

## Limitations
- Accuracy depends on training data
- Performance varies with video quality
- Requires clear face detection in frames

# Deepfake Detection Results

## Experimental Setup
- **Dataset**: Deepfake TIMIT
- **Classification Thresholds**:
  - Real: ≤ 0.2
  - Uncertain: 0.2 - 0.8
  - Fake: ≥ 0.8

## Model Performance Comparison

| Model | PROBABLY FAKE | UNCERTAIN | PROBABLY REAL | Total | Accuracy (%) | Precision | Recall | TP | FP | FN |
|-------|------|-----------|------|-------|--------------|-----------|--------|----|----|-----|
| **EfficientNetAutoAttB4** | 520 | 98 | 22 | 640 | 81.25 | 0.81 | 0.81 | 520 | 120 | 120 |
| **EfficientNetAutoAttB4ST** | 589 | 49 | 2 | 640 | 92.03 | 0.92 | 0.92 | 589 | 51 | 51 |
| **EfficientNetB4** | 545 | 87 | 8 | 640 | 85.16 | 0.87 | 0.85 | 545 | 95 | 95 |
| **EfficientNetB4ST** | 560 | 73 | 7 | 640 | 87.50 | 0.88 | 0.88 | 560 | 80 | 80 |

Calculation notes:
- Since the true label is FAKE, all samples NOT classified as FAKE are considered false positives (FP) and false negatives (FN)
- TP (True Positives) = Number of correctly classified FAKE samples
- FP = Number of samples incorrectly classified as UNCERTAIN or PROBABLY REAL
- FN = Number of samples incorrectly classified as UNCERTAIN or PROBABLY REAL
- Precision = TP / (TP + FP)
- Recall = TP / Total number of samples

The EfficientNetAutoAttB4ST model shows the best performance with the highest precision and recall of 0.92.

## Detailed Analysis

### Best Performing Model
**EfficientNetAutoAttB4ST** emerged as the top-performing model with:
- Highest accuracy: 92.03%
- Lowest uncertainty rate
- Most confident fake video detection

## Experimental Setup
- **Dataset**: SDFVD
- **Classification Thresholds**:
  - Real: ≤ 0.2
  - Uncertain: 0.2 - 0.8
  - Fake: ≥ 0.8

| Video Type | PROBABLY FAKE | UNCERTAIN | PROBABLY REAL | Total | Accuracy (%) |
|------------|------|-----------|------|-------|--------------|
| **Fake Videos** | 4 | 21 | 28 | 53 | 7.55 |
| **Real Videos** | 1 | 24 | 28 | 53 | 52.83 |

## Key Observations
1. The self-attention variants (ST models) consistently outperformed their base counterparts
2. High uncertainty rates
3. Uncertainty boundaries could be modified to improve results. Dependant on user requirements.

## Inference Times  
- **EfficientNetAutoAttB4ST**: Processes 1 second of video in approximately 1 second on an M1 Mac CPU.  

## Recommended Model
**EfficientNetAutoAttB4ST** is recommended for future deepfake detection tasks due to its superior performance and low uncertainty rate.

# Instructions to Generate Metrics

1. **Dataset**: You need one of the following datasets:
   - **DeepFakeTIMIT** (Lower Quality and Higher Quality)
   - **SDFVD** (videos_fake and videos_real)

   Download the dataset from the appropriate source and extract it to a local directory.

## Instructions

### 1. Setup for DeepFakeTIMIT Dataset
   - Place the **Lower Quality** videos in one directory and the **Higher Quality** videos in another.
   - Run the script separately for each quality level and collate the results afterward.

### 2. Running the Script
   - Run the script using:
     ```bash
     python evalpipeline.py
     ```
   - When prompted, enter the path to the directory containing the video files for the dataset (either Lower Quality or Higher Quality for DeepFakeTIMIT, or videos_fake and videos_real for SDFVD).

### 3. Script Behavior
   - The script will:
     1. Search for video files in the specified directory (including subdirectories).
     2. Process each video using the detection models: `EfficientNetB4`, `EfficientNetB4ST`, `EfficientNetAutoAttB4`, and `EfficientNetAutoAttB4ST`.
     3. Generate a results CSV file for each model, saved in the same directory as the script (e.g., `deepfake_results_EfficientNetB4.csv`).

### 4. Collating Results for DeepFakeTIMIT
   - After processing both the **Lower Quality** and **Higher Quality** datasets:
     - Combine the respective CSV files using a tool like pandas:
       ```python
       import pandas as pd

       low_quality_results = pd.read_csv('deepfake_results_EfficientNetB4.csv')  # Update file name as necessary
       high_quality_results = pd.read_csv('deepfake_results_EfficientNetB4_HQ.csv')

       combined_results = pd.concat([low_quality_results, high_quality_results])
       combined_results.to_csv('deepfake_combined_results.csv', index=False)
       ```
     - Review the combined results.

### 5. Output
   - For each model, the script will print the number of predictions in the following categories:
     - PROBABLY Real
     - PROBABLY Fake
     - Uncertain

### Notes
   - Adjust the `real_threshold` and `fake_threshold` parameters in the `run_detection` call if needed.
   - Ensure the `detect_deepfakes` module and its dependencies are correctly configured. Refer to its documentation for details.
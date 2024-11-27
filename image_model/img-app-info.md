# Image Deepfake Detector

## Overview
This application is a machine learning-powered deepfake detection tool that analyzes image files to determine whether they contain manipulated/generated (fake) or real content. It uses a ML architecture using a binary NN backbone to perform image classification.

## Key Components
1. **Server for Image Classifier (`model_server.py`)**: 
   - Creates a Flask-based server to host the Image classifier model
   - Contains code to work with RescueBox client.
   - API can work with path to directory containing images and creates a CSV file containing output.
   - Applies the appropriating pre-processing steps and runs the model on collection of images.

2. **Testing code (`test.py`)**: 
   - Can be used to test datasets. 
   - Assumes all fake data files have "F" in the name and uses this to assige labels.
   - Outputs metrics 

## Setup

### 1. Clone the required repositories and restructure code
Note the BNext repository needs to be cloned into the binary_deepfake_detection one. 
```bash
git clone https://github.com/aravadikesh/DeepFakeDetector.git
git clone https://github.com/fedeloper/binary_deepfake_detection.git
cd .\binary_deepfake_detection\
git clone https://github.com/hpi-xnor/BNext.git

```

Now move the files under `image_model` folder to the binary_deepfake_detection folder, replacing existing files if needed. Your structure should be like - 

```
binary_deepfake_detection\
│
├── BNext\                  # ML backbone code
├── pretrained\             # pretrained model ckpts
├── requirements.txt        # Python package dependencies
├── model.py                # code for ML model           
├── model_server.py         # Flask-ML server code   
├── test.py                 # Testing code
├── sim_data.py             # Data reading and preprocessing  
├── img-app-info.md         # Info 
└── client.py               # Flask-ML cli for server
```

### 2. Download weights and checkpoint

- Download the model backbone weights from [here](https://drive.google.com/file/d/1xyKnA6SsG4ZpguNQQrB6Yz-J5dzXYfKE/view), prefix the name with `middle_`, and move them to the `pretrained` folder. You should have `pretrained\middle_checkpoint.pth.tar`. Other backbone weights available [here](https://github.com/hpi-xnor/BNext/tree/main?tab=readme-ov-file).

- Create a `weights\` folder in the main program folder (`binary_deepfake_detection\`). Download the model checkpoint from [here](https://drive.google.com/file/d/16c5xIDvwN3DUD6JbO_cl7aj_xrijezWs/view?usp=drive_link) and place the downloaded file into the weights folder. Other checkpoints (trained on other datasets/using different backbones) can be found [here](https://drive.google.com/drive/folders/1rYtfozcq5eXK1a8tP8ouXrBFZs1e72dV).


### 3. Install Dependencies

`conda` and `pip` required for next steps. We create a new conda env with python 3.12 and install required libs.

```bash
conda create --name img_dfd
conda activate img_dfd
conda install python=3.12
pip install -r requirements.txt
```

### Usage

Now you can run `python model_server.py` and use the RescueBox GUI for interacting with the server.


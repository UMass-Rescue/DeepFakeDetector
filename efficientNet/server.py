
import argparse
import os
from typing import TypedDict
import pandas as pd
import json
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    BatchFileInput,
    BatchFileResponse,
    FileResponse, ResponseType, FileType,
    InputSchema,
    InputType,
    ResponseBody,
    TaskSchema,
    DirectoryInput,
    EnumParameterDescriptor,
    EnumVal,
    InputSchema,
    InputType,
    ParameterSchema,
    ResponseBody,
    TaskSchema,
    FileResponse,
    FileType,
    FloatParameterDescriptor
)
from detect_deepfakes import run_detection

server = MLServer(__name__)

server.add_app_metadata(name="EfficientNet Video DeepFake Detector", author="UMass Rescue", version="0.1.0", info=load_file_as_string("app_info.md"))

def create_deepfake_detection_task_schema() -> TaskSchema:
    return TaskSchema(
        inputs=[
            InputSchema(
            key="video_paths",
            label="Video Paths",
            input_type=InputType.BATCHFILE,
            ),
            InputSchema(
            key="output_directory",
            label="Output Directory",
            input_type=InputType.DIRECTORY,
            ),
        ],
        parameters=[
            ParameterSchema(
            key="model",
            label="Model",
            value=EnumParameterDescriptor(
                enum_vals=[
                EnumVal(key="EfficientNetB4", label="EfficientNetB4"),
                EnumVal(key="EfficientNetB4ST", label="EfficientNetB4ST"),
                EnumVal(key="EfficientNetAutoAttB4", label="EfficientNetAutoAttB4"),
                EnumVal(key="EfficientNetAutoAttB4ST", label="EfficientNetAutoAttB4ST"),
                ],
                default="EfficientNetAutoAttB4ST",
            )
            ),
            ParameterSchema(
                key="real_threshold",
                label="Real Threshold",
                value=FloatParameterDescriptor(default=0.2),
                ),
            ParameterSchema(
                key="fake_threshold",
                label="Fake Threshold",
                value= FloatParameterDescriptor(default=0.8),
                ),
        ],
        )

# Define input types
class DeepfakeDetectionInputs(TypedDict):
    video_paths: BatchFileInput  # Accepts multiple video file paths
    output_directory: DirectoryInput  # Accepts a directory path

class DeepfakeDetectionParameters(TypedDict):
    model: str
    real_threshold: float
    fake_threshold: float

@server.route("/detect_deepfake", task_schema_func=create_deepfake_detection_task_schema , short_title="Deepfake Detection", order=0)
def detect_deepfake(inputs: DeepfakeDetectionInputs, parameters: DeepfakeDetectionParameters):
    video_files = [file.path for file in inputs['video_paths'].files]
    output_dir = inputs['output_directory'].path

    # Validate threshold values
    real_threshold = parameters['real_threshold']
    fake_threshold = parameters['fake_threshold']
    if not (0 <= real_threshold <= 1):
        raise ValueError("Real threshold must be between 0 and 1.")
    if not (0 <= fake_threshold <= 1):
        raise ValueError("Fake threshold must be between 0 and 1.")
    if real_threshold >= fake_threshold:
        raise ValueError("Real threshold must be less than fake threshold.")

    # Run detection and save to CSV
    results_df = run_detection(video_files, model=parameters['model'], real_threshold=parameters['real_threshold'], fake_threshold=parameters['fake_threshold']) 
    output_csv = os.path.join(output_dir, "deepfake_results.csv")
    results_df.to_csv(output_csv, index=False)
    
    return ResponseBody(root=FileResponse(output_type=ResponseType.FILE, file_type=FileType.CSV, path=output_csv, title="Detection Results", subtitle="Deepfake detection CSV results"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a server.")
    parser.add_argument(
        "--port", type=int, help="Port number to run the server", default=5000
    )
    args = parser.parse_args()
    server.run(port=args.port)

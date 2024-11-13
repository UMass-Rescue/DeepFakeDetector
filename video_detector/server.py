from typing import TypedDict
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
    BatchDirectoryInput,
    EnumParameterDescriptor,
    EnumVal,
    InputSchema,
    InputType,
    ParameterSchema,
    ResponseBody,
    TaskSchema,
    MarkdownResponse,
    TextParameterDescriptor,
    TextInput,
    BatchFileResponse,
    IntParameterDescriptor,
    FileResponse,
    FileType,
)
from video_evaluator import VideoEvaluator  
import pdb

# Initialize Flask-ML server
server = MLServer(__name__)

server.add_app_metadata(
    name="Video DeepFake Detector",
    author="UMass Rescue",
    version="0.1.0",
    info=load_file_as_string("app_info.md"),
)

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
                key="cuda",
                label="CUDA",
                value=EnumParameterDescriptor(
                    enum_vals=[
                        EnumVal(key="True", label="True"),
                        EnumVal(key="False", label="False"),
                    ],
                    default="False",
                )
            ),
        ],
    )

# Define input types
class DeepfakeDetectionInputs(TypedDict):
    video_paths: BatchFileInput  # Accepts multiple video file paths
    output_directory: DirectoryInput  # Accepts a directory path

class DeepfakeDetectionParameters(TypedDict):
    cuda: str

@server.route(
    "/detect_deepfake",
    task_schema_func=create_deepfake_detection_task_schema,
    short_title="Deepfake Detection",
    order=0
)
def detect_deepfake(inputs: DeepfakeDetectionInputs, parameters: DeepfakeDetectionParameters) -> ResponseBody:    
    # Initialize the VideoEvaluator with model and output paths
    model_path = "path/to/your/model.pth"  # Path to the pre-trained model
    output_path = inputs["output_directory"].path # Directory to save processed videos
    cuda_flag = parameters["cuda"] == "True"
    evaluator = VideoEvaluator(model_path=model_path, output_path=output_path, cuda=cuda_flag)
    
    # Process each video file path and store the output paths
    output_paths = []
    for video_path in inputs['video_paths'].files:
        # pdb.set_trace()
        # Run the evaluation
        processed_video_path = evaluator.evaluate_video(video_path.path)

        if processed_video_path is not None:
            # Construct FileResponse with required fields
            output_paths.append(
                FileResponse(
                    output_type=ResponseType.FILE,
                    path=processed_video_path,
                    title=f"Processed {video_path}",
                    file_type=FileType.VIDEO 
                )
            )
        else:
            # Handle the case where processed_video_path is None
            print(f"Failed to process video: {video_path.path}")
            
    # Return the processed file paths as a BatchFileResponse
    return ResponseBody(root=BatchFileResponse(files=output_paths))

if __name__ == "__main__":
    # Run the server
    server.run()

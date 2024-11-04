from typing import TypedDict
from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.models import (
    BatchFileInput,
    BatchFileResponse,
    FileResponse, ResponseType, FileType,
    InputSchema,
    InputType,
    ResponseBody,
    TaskSchema,
)
from video_evaluator import VideoEvaluator  
import pdb

# Initialize Flask-ML server
server = MLServer(__name__)

# Define input types
class DeepfakeDetectionInputs(TypedDict):
    video_paths: BatchFileInput  # Accepts multiple video file paths

class DeepfakeDetectionParameters(TypedDict):
    output_path: str

# def create_deepfake_detection_task_schema() -> TaskSchema:
#     # Define the input schema for video paths
#     input_schema = InputSchema(
#         key="video_paths",
#         label="Videos to Process",
#         input_type=InputType.BATCHFILE
#     )
#     # Define the parameter schema for detection threshold
#     # parameter_schema = ParameterSchema(
#     #     key="detection_threshold",
#     #     label="Detection Threshold",
#     #     type="number",
#     #     default=0.5
#     # )
#     parameter_schema = {}
#     return TaskSchema(
#         inputs=[input_schema],
#         parameters=[parameter_schema]
#     )

@server.route(
    "/detect_deepfake",
    # task_schema_func=create_deepfake_detection_task_schema,
    # short_title="Deepfake Detection",
    # order=0
)
def detect_deepfake(inputs: DeepfakeDetectionInputs, parameters: DeepfakeDetectionParameters) -> ResponseBody:    
    # Initialize the VideoEvaluator with model and output paths
    model_path = "path/to/your/model.pth"  # Path to the pre-trained model
    output_path = parameters['output_path']        # Directory to save processed videos
    evaluator = VideoEvaluator(model_path=model_path, output_path=output_path, cuda=False)
    
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

#python detect_from_video.py -i /Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/SDFVD/videos_fake/vs1.mp4 -o /Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/results

if __name__ == "__main__":
    # Run the server
    server.run()

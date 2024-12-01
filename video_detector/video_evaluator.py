import os
import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from os.path import join
from PIL import Image as pil_image
from tqdm import tqdm
from network.models import model_selection, return_pytorch04_xception
from dataset.transform import xception_default_data_transforms
import pdb
import json

class VideoEvaluator:
    def __init__(self, model_name=None, output_path='.', cuda=False, model_path=None):
        self.output_path = output_path
        self.cuda = torch.cuda.is_available() if cuda else False
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Load model
        if model_name is None:
            self.model, *_ = model_selection(modelname='xception', num_out_classes=2, model_path=model_path)
        elif model_name == 'xception':
            self.model, *_ = model_selection(modelname='xception', num_out_classes=2, model_path=model_path)
        elif model_name == 'resnet18':
            self.model, *_ = model_selection(modelname='resnet18', num_out_classes=2)
        elif model_name == 'resnet50':
            self.model, *_ = model_selection(modelname='resnet50', num_out_classes=2)

        if self.cuda:
            self.model = self.model.cuda()

    def get_boundingbox(self, face, width, height, scale=1.3, minsize=None):
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        size_bb = int(max(x2 - x1, y2 - y1) * scale)
        if minsize and size_bb < minsize:
            size_bb = minsize
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        x1, y1 = max(int(center_x - size_bb // 2), 0), max(int(center_y - size_bb // 2), 0)
        size_bb = min(width - x1, size_bb)
        size_bb = min(height - y1, size_bb)
        return x1, y1, size_bb

    def preprocess_image(self, image):
        """
        Preprocess the input image for model prediction.
        
        Args:
            image (numpy.ndarray): The input image in BGR format.
        
        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocess = xception_default_data_transforms['test']
        preprocessed_image = preprocess(pil_image.fromarray(image)).unsqueeze(0)
        if self.cuda:
            preprocessed_image = preprocessed_image.cuda()
        return preprocessed_image

    def predict_with_model(self, image):
        preprocessed_image = self.preprocess_image(image)
        output = self.model(preprocessed_image)
        output = nn.Softmax(dim=1)(output)
        _, prediction = torch.max(output, 1)
        return int(prediction.cpu().numpy()), output

    def evaluate_video(self, video_path, start_frame=0, end_frame=None, output_mode='video', verbose=False):
        """
        Evaluate a video for deepfake detection with multiple output modes.
        
        Args:
            video_path (str): Path to input video
            start_frame (int): Starting frame for processing
            end_frame (int): Ending frame for processing
            output_mode (str): Either 'video' for processed video output or 'json' for detection results
            verbose (bool): If True, includes detailed frame-by-frame analysis in JSON output
        
        Returns:
            Union[str, dict]: Either the path to processed video or path to JSON results
        """
        print(f'Starting: {video_path}')
        
        # Setup for video input
        reader = cv2.VideoCapture(video_path)
        num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = reader.get(cv2.CAP_PROP_FPS)
        frame_num = 0
        
        # Initialize video writer if in video mode
        writer = None
        processed_video_path = None
        json_output_path = None
        
        # Create output paths
        base_filename = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(self.output_path, exist_ok=True)
        
        if output_mode == 'video':
            processed_video_path = os.path.join(self.output_path, f"{base_filename}_processed.avi")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        else:
            json_output_path = os.path.join(self.output_path, f"{base_filename}_results.json")

        # Initialize results tracking
        total_predictions = 0
        sum_predictions = 0
        sum_confidence = np.array([0.0, 0.0])  # For averaging confidence scores
        frames_with_faces = 0
        
        # Initialize results dictionary for JSON mode
        json_results = {
            'input_path': video_path,
            'frames_analyzed': 0,
            'frames_with_faces': 0,
        }
        if verbose:
            json_results['frames'] = []

        pbar = tqdm(total=(end_frame - start_frame) if end_frame else num_frames)
        while reader.isOpened():
            ret, image = reader.read()
            if not ret or (end_frame and frame_num >= end_frame):
                break
            frame_num += 1
            if frame_num < start_frame:
                continue
            pbar.update(1)

            if output_mode == 'video' and writer is None:
                writer = cv2.VideoWriter(processed_video_path, fourcc, fps, 
                                       (image.shape[1], image.shape[0]))
            
            # Detect and process faces in the frame
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray, 1)
            
            if verbose:
                frame_results = {
                    'frame_number': frame_num,
                    'faces': []
                }
            
            if faces:
                frames_with_faces += 1
                face = faces[0]
                x, y, size = self.get_boundingbox(face, image.shape[1], image.shape[0])
                cropped_face = image[y:y+size, x:x+size]
                prediction, output = self.predict_with_model(cropped_face)
                
                # Update running totals
                total_predictions += 1
                sum_predictions += prediction
                output_np = output.detach().cpu().numpy()
                sum_confidence += output_np[0]
                
                if verbose:
                    # Store detailed face detection results
                    face_result = {
                        'bbox': {'x': x, 'y': y, 'width': face.width(), 'height': face.height()},
                        'prediction': int(prediction),
                        'confidence': output.tolist(),
                        'label': 'fake' if prediction == 1 else 'real'
                    }
                    frame_results['faces'].append(face_result)
                
                if output_mode == 'video':
                    # Annotate frame
                    label = 'fake' if prediction == 1 else 'real'
                    color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                    cv2.putText(image, f"{output.tolist()} => {label}", 
                              (x, y + face.height() + 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.rectangle(image, (x, y), (face.right(), face.bottom()), 
                                color, 2)

            if output_mode == 'video':
                writer.write(image)
            elif verbose:
                json_results['frames'].append(frame_results)
                
        pbar.close()
        reader.release()
        if writer is not None:
            writer.release()

        if output_mode == 'video':
            print(f'Finished! Output saved under {processed_video_path}')
            return processed_video_path
        else:
            # Calculate final predictions and confidence
            if total_predictions > 0:
                avg_prediction = sum_predictions / total_predictions
                avg_confidence = sum_confidence / total_predictions
                
                # Determine final label
                final_label = 'fake' if avg_prediction >= 0.5 else 'real'
                
                # Update JSON results with summary
                json_results.update({
                    'frames_analyzed': frame_num - start_frame,
                    'frames_with_faces': frames_with_faces,
                    'final_label': final_label,
                    'confidence_scores': {
                        'real': float(avg_confidence[0]),
                        'fake': float(avg_confidence[1])
                    },
                    'average_prediction': float(avg_prediction)
                })
            else:
                json_results.update({
                    'frames_analyzed': frame_num - start_frame,
                    'frames_with_faces': 0,
                    'final_label': 'no_faces_detected',
                    'confidence_scores': {
                        'real': 0.0,
                        'fake': 0.0
                    },
                    'average_prediction': 0.0
                })

            # # Write JSON results to file
            # with open(json_output_path, 'w', encoding='utf-8') as f:
            #     json.dump(json_results, f, indent=2)
            # print(f'Finished! JSON results saved under {json_output_path}')
            # return json_output_path
            return json_results
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-i', type=str, required=True)
    parser.add_argument('--model_path', '-m', type=str, default=None)
    parser.add_argument('--output_path', '-o', type=str, default='.')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=None)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--model', '-M', type=str, default='xception')
    args = parser.parse_args()
    
    evaluatorX = VideoEvaluator(model_name='xception', model_path=args.model_path, output_path=args.output_path, cuda=args.cuda)
    evaluator18 = VideoEvaluator(model_name='resnet18', model_path=None, output_path=args.output_path, cuda=args.cuda)
    evaluator50 = VideoEvaluator(model_name='resnet50', model_path=None, output_path=args.output_path, cuda=args.cuda)

    dirs = ['/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/SDFVD/videos_fake', '/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/SDFVD/videos_real']

    # for dir in dirs:
    #     if os.path.isdir(dir):
    #         dirName = os.path.basename(dir)
    #         X_results = []
    #         res18_results = []
    #         res50_results = []
            
    #         for video in os.listdir(args.video_path):
    #             video_path = join(args.video_path, video)
    #             print(f'Processing video: {video_path}')
    #             resultX = evaluatorX.evaluate_video(video_path, args.start_frame, args.end_frame, output_mode='video')
    #             # result18 = evaluator18.evaluate_video(video_path, args.start_frame, args.end_frame, output_mode='json')
    #             # result50 = evaluator50.evaluate_video(video_path, args.start_frame, args.end_frame, output_mode='json')

    #             X_results.append(resultX)
    #             # res18_results.append(result18)
    #             # res50_results.append(result50)
            
    #         # Save all results to a single JSON file
    #         # output_file = os.path.join(args.output_path, dirName + 'X_results.json')
    #         # with open(output_file, 'w', encoding='utf-8') as f:
    #         #     json.dump(X_results, f, indent=2)

    #         # output_file = os.path.join(args.output_path, dirName + '18_results.json')
    #         # with open(output_file, 'w', encoding='utf-8') as f:
    #         #     json.dump(res18_results, f, indent=2)

    #         # output_file = os.path.join(args.output_path, dirName + 'res50_results.json')
    #         # with open(output_file, 'w', encoding='utf-8') as f:
    #         #     json.dump(res50_results, f, indent=2)

    #         # print(f'All results saved under {output_file}')
    #         print(X_results)
    # # else:
    # Average results over 5 runs
    results = []
    for _ in range(5):
        res = evaluatorX.evaluate_video(args.video_path, args.start_frame, args.end_frame, output_mode='json')
        results.append(res)

    # Calculate average results
    avg_results = {
        'frames_analyzed': np.mean([r['frames_analyzed'] for r in results]),
        'frames_with_faces': np.mean([r['frames_with_faces'] for r in results]),
        'confidence_scores': {
            'real': np.mean([r['confidence_scores']['real'] for r in results]),
            'fake': np.mean([r['confidence_scores']['fake'] for r in results])
        },
        'average_prediction': np.mean([r['average_prediction'] for r in results]),
        'final_label': 'fake' if np.mean([r['average_prediction'] for r in results]) >= 0.5 else 'real'
    }

    print(avg_results)
    # res = evaluatorX.evaluate_video(args.video_path, args.start_frame, args.end_frame, output_mode='json')
    # evaluator18.evaluate_video(args.video_path, args.start_frame, args.end_frame, output_mode='video')
    #res = evaluator50.evaluate_video(args.video_path, args.start_frame, args.end_frame, output_mode='json')
    print(res)

# python video_evaluator.py -i '/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/DeepfakeTIMIT/higher_quality/fadg0/sa1-video-fram1.avi' 
# -o '/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/results' 
# -m '/Users/aravadikesh/Documents/GitHub/DeepFakeDetector/video_detector/weights/xception-b5690688.pth'




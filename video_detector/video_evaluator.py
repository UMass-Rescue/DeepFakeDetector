import os
import cv2
import dlib
import torch
import torch.nn as nn
from os.path import join
from PIL import Image as pil_image
from tqdm import tqdm
from network.models import model_selection, return_pytorch04_xception
from dataset.transform import xception_default_data_transforms
import pdb

class VideoEvaluator:
    def __init__(self, model_path=None, output_path='.', cuda=True):
        self.output_path = output_path
        self.cuda = cuda
        self.face_detector = dlib.get_frontal_face_detector()
        
        # Load model
        self.model, *_ = model_selection(modelname='xception', num_out_classes=2)
        if model_path:
            self.model = return_pytorch04_xception()
            print(f'Model found in {model_path}')
        else:
            print('No model found, initializing random model.')
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

    # def evaluate_video(self, video_path, start_frame=0, end_frame=None):
    #     print(f'Starting: {video_path}')
    #     # pdb.set_trace()
    #     reader = cv2.VideoCapture(video_path)
    #     video_fn = f"{os.path.splitext(os.path.basename(video_path))[0]}.avi"
    #     # pdb.set_trace()
    #     os.makedirs(self.output_path, exist_ok=True)
        
    #     fourcc, fps, num_frames = cv2.VideoWriter_fourcc(*'MJPG'), reader.get(cv2.CAP_PROP_FPS), int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    #     writer, frame_num = None, 0

    #     pbar = tqdm(total=(end_frame - start_frame) if end_frame else num_frames)
    #     while reader.isOpened():
    #         ret, image = reader.read()
    #         if not ret or (end_frame and frame_num >= end_frame):
    #             break
    #         frame_num += 1
    #         if frame_num < start_frame:
    #             continue
    #         pbar.update(1)

    #         if writer is None:
    #             writer = cv2.VideoWriter(join(self.output_path, video_fn), fourcc, fps, (image.shape[1], image.shape[0]))
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         faces = self.face_detector(gray, 1)
    #         if faces:
    #             face = faces[0]
    #             x, y, size = self.get_boundingbox(face, image.shape[1], image.shape[0])
    #             cropped_face = image[y:y+size, x:x+size]
    #             prediction, output = self.predict_with_model(cropped_face)
                
    #             label = 'fake' if prediction == 1 else 'real'
    #             color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
    #             cv2.putText(image, f"{output.tolist()}=>{label}", (x, y + face.height() + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    #             cv2.rectangle(image, (x, y), (face.right(), face.bottom()), color, 2)

    #         writer.write(image)
    #     pbar.close()
    #     writer.release()
    #     print(f'Finished! Output saved under {self.output_path}')
    #     return self.output_path
    def evaluate_video(self, video_path, start_frame=0, end_frame=None):
        print(f'Starting: {video_path}')
        
        # Setup for video input and output paths
        reader = cv2.VideoCapture(video_path)
        video_fn = f"{os.path.splitext(os.path.basename(video_path))[0]}_processed.avi"
        processed_video_path = os.path.join(self.output_path, video_fn)
        os.makedirs(self.output_path, exist_ok=True)
        
        # Video writer setup
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        fps = reader.get(cv2.CAP_PROP_FPS)
        num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = None
        frame_num = 0

        pbar = tqdm(total=(end_frame - start_frame) if end_frame else num_frames)
        while reader.isOpened():
            ret, image = reader.read()
            if not ret or (end_frame and frame_num >= end_frame):
                break
            frame_num += 1
            if frame_num < start_frame:
                continue
            pbar.update(1)

            if writer is None:
                writer = cv2.VideoWriter(processed_video_path, fourcc, fps, (image.shape[1], image.shape[0]))
            
            # Detect and process faces in the frame
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray, 1)
            if faces:
                face = faces[0]
                x, y, size = self.get_boundingbox(face, image.shape[1], image.shape[0])
                cropped_face = image[y:y+size, x:x+size]
                prediction, output = self.predict_with_model(cropped_face)
                
                # Annotate frame
                label = 'fake' if prediction == 1 else 'real'
                color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
                cv2.putText(image, f"{output.tolist()} => {label}", (x, y + face.height() + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.rectangle(image, (x, y), (face.right(), face.bottom()), color, 2)

            writer.write(image)
        pbar.close()
        reader.release()
        if writer is not None:
            writer.release()

        print(f'Finished! Output saved under {processed_video_path}')
        
        # Return the path to the processed video file
        return processed_video_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', '-i', type=str, required=True)
    parser.add_argument('--model_path', '-m', type=str, default=None)
    parser.add_argument('--output_path', '-o', type=str, default='.')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=None)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    
    evaluator = VideoEvaluator(args.model_path, args.output_path, args.cuda)
    if os.path.isdir(args.video_path):
        for video in os.listdir(args.video_path):
            evaluator.evaluate_video(join(args.video_path, video), args.start_frame, args.end_frame)
    else:
        evaluator.evaluate_video(args.video_path, args.start_frame, args.end_frame)


import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from scipy.special import expit

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet, weights
from isplutils import utils
from tqdm import tqdm

class DeepfakeDetector:
    def __init__(self, net_model: str = 'EfficientNetAutoAttB4ST', train_db: str = 'DFDC', real_threshold: float = 0.2, fake_threshold: float = 0.75):
        self.real_threshold = real_threshold
        self.fake_threshold = fake_threshold
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        model_url = weights.weight_url[f'{net_model}_{train_db}']
        self.net = getattr(fornet, net_model)().eval().to(self.device)
        self.net.load_state_dict(torch.hub.load_state_dict_from_url(model_url, map_location=self.device, check_hash=True))
        
        self.transf = utils.get_transformer('scale', 224, self.net.get_normalizer(), train=False)
        
        self.facedet = BlazeFace().to(self.device)
        self.facedet.load_weights("blazeface/blazeface.pth")
        self.facedet.load_anchors("blazeface/anchors.npy")
        
        videoreader = VideoReader(verbose=False)
        video_read_fn = lambda x: videoreader.read_frames(x, num_frames=32)
        self.face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=self.facedet)

    def process_video(self, video_path: str) -> Dict[str, float]:
        try:
            vid_faces = self.face_extractor.process_video(video_path)
            faces_t = torch.stack([self.transf(image=frame['faces'][0])['image'] for frame in vid_faces if len(frame['faces'])])
            
            with torch.no_grad():
                faces_pred = self.net(faces_t.to(self.device)).cpu().numpy().flatten()
            
            prob = expit(faces_pred).mean()
            
            if prob <= self.real_threshold:
                label = 'PROBABLY REAL'
            elif prob >= self.fake_threshold:
                label = 'PROBABLY FAKE'
            else:
                label = 'UNCERTAIN'
            return {'video_path': os.path.basename(video_path), 'deepfake_probability': prob, 'prediction': label} 
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return {'video_path': video_path, 'deepfake_probability': None, 'prediction': 'ERROR'}

def run_detection(video_paths: List[str], model: str, real_threshold: float, fake_threshold: float) -> pd.DataFrame:
    detector = DeepfakeDetector(net_model=model, real_threshold=real_threshold, fake_threshold=fake_threshold)
    results = []
    for video in tqdm(video_paths):
        results.append(detector.process_video(video))
    return pd.DataFrame(results)

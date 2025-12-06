import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from collections import deque
import os
from ament_index_python.packages import get_package_share_directory

# Owen's custom CNN for hand "pinch" classification.

class CustomFingerPinch:
    
    def __init__(self, model_path='hand_gesture_model.pth', 
                 img_size=(224, 224),
                 smoothing_window=10,
                 threshold=0.5):
        
        self.img_size = img_size
        self.class_names = ['UNPINCHED', 'PINCHED']
        self.model = None
        self.prediction_history = deque(maxlen=smoothing_window)
        self.threshold = threshold
        
        # Resolve model path - try as absolute path first, then in package weights directory
        if not os.path.isabs(model_path) or not os.path.exists(model_path):
            try:
                package_share_dir = get_package_share_directory('unified_vision_node')
                weights_path = os.path.join(package_share_dir, 'weights', model_path)
                if os.path.exists(weights_path):
                    model_path = weights_path
            except Exception:
                # Fall back to relative path from this file
                weights_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'weights')
                weights_path = os.path.join(weights_dir, model_path)
                if os.path.exists(weights_path):
                    model_path = weights_path
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_model(model_path)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        base_model = models.efficientnet_b0(weights=None)
        num_features = base_model.classifier[1].in_features
        
        base_model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        base_model.load_state_dict(checkpoint['model_state_dict'])
        base_model.eval()
        
        self.model = base_model.to(self.device)
        
        if 'img_size' in checkpoint:
            self.img_size = checkpoint['img_size']
    
    def __call__(self, rgb_image, depth_image, camera_info):
        preprocessed = self.preprocess_frame(rgb_image)
        preprocessed = preprocessed.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(preprocessed).cpu().numpy()[0][0]
        
        # temporal smoothing
        self.prediction_history.append(prediction)
        smoothed_prediction = np.mean(self.prediction_history)
        
        is_pinching = smoothed_prediction > self.threshold
        
        # create annotated image
        annotated_image = rgb_image.copy()
        
        # add text overlay
        text = "PINCHING" if is_pinching else "OPEN"
        color = (0, 255, 0) if is_pinching else (255, 0, 0)
        conf_text = f"{smoothed_prediction:.2f}"
        
        cv2.putText(annotated_image, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.putText(annotated_image, conf_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        output = {
            'is_pinching': bool(is_pinching),
            'hand_landmarks': None,  # cnn model doesn't provide landmarks
            'palm_centroid_pixel': None,  # no landmarks available
            'normalized_spread': float(smoothed_prediction),  # use confidence as spread
            'annotated_image': annotated_image
        }
        
        return output
    
    def preprocess_frame(self, frame):
        # resize and convert bgr to rgb
        img = cv2.resize(frame, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # apply normalization
        img = self.transform(img)
        
        # add batch dimension
        img = img.unsqueeze(0)
        
        return img


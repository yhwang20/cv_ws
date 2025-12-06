import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Union, Optional
import numpy as np
import cv2

from .config import DetectorConfig


class HandLandmarkDetector:
    
    def __init__(self, config: Union[DetectorConfig, str]):
        if isinstance(config, str):
            config = DetectorConfig(model_path=config)
        
        self.config = config
        self._detector = self._create_detector()
    
    def _create_detector(self) -> vision.HandLandmarker:
        base_options = python.BaseOptions(model_asset_path=self.config.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.min_hand_detection_confidence,
            min_hand_presence_confidence=self.config.min_hand_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        return vision.HandLandmarker.create_from_options(options)
    
    def detect(self, image: Union[np.ndarray, str, mp.Image]) -> vision.HandLandmarkerResult:
        mp_image = self._to_mp_image(image)
        detection_result = self._detector.detect(mp_image)
        return detection_result
    
    def _to_mp_image(self, image: Union[np.ndarray, str, mp.Image]) -> mp.Image:
        if isinstance(image, mp.Image):
            return image
        elif isinstance(image, str):
            return mp.Image.create_from_file(image)
        elif isinstance(image, np.ndarray):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        else:
            raise TypeError(f"unsupported image type: {type(image)}")
    
    def detect_from_file(self, image_path: str) -> vision.HandLandmarkerResult:
        return self.detect(image_path)
    
    def detect_from_array(self, image_array: np.ndarray) -> vision.HandLandmarkerResult:
        return self.detect(image_array)

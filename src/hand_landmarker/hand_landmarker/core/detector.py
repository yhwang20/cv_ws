"""Core hand landmark detection module."""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Union, Optional
import numpy as np
import cv2

from .config import DetectorConfig


class HandLandmarkDetector:
    """Hand landmark detector using MediaPipe.
    
    This class provides an interface for detecting hand landmarks in images.
    It wraps the MediaPipe HandLandmarker API for easier use.
    
    Example:
        >>> detector = HandLandmarkDetector("hand_landmarker.task")
        >>> image = cv2.imread("image.jpg")
        >>> result = detector.detect(image)
        >>> annotated = detector.draw_landmarks(image, result)
    """
    
    def __init__(self, config: Union[DetectorConfig, str]):
        """Initialize the hand landmark detector.
        
        Args:
            config: Either a DetectorConfig object or a string path to the model file.
                   If a string is provided, default configuration will be used.
        """
        if isinstance(config, str):
            config = DetectorConfig(model_path=config)
        
        self.config = config
        self._detector = self._create_detector()
    
    def _create_detector(self) -> vision.HandLandmarker:
        """Create and configure the MediaPipe HandLandmarker."""
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
        """Detect hand landmarks in an image.
        
        Args:
            image: Input image. Can be:
                  - numpy array (BGR or RGB format)
                  - string path to image file
                  - MediaPipe Image object
        
        Returns:
            HandLandmarkerResult containing detected hand landmarks and handedness.
        """
        # Convert input to MediaPipe Image format
        mp_image = self._to_mp_image(image)
        
        # Perform detection
        detection_result = self._detector.detect(mp_image)
        
        return detection_result
    
    def _to_mp_image(self, image: Union[np.ndarray, str, mp.Image]) -> mp.Image:
        """Convert various image formats to MediaPipe Image.
        
        Args:
            image: Input image in various formats
        
        Returns:
            MediaPipe Image object
        """
        if isinstance(image, mp.Image):
            return image
        elif isinstance(image, str):
            # File path
            return mp.Image.create_from_file(image)
        elif isinstance(image, np.ndarray):
            # Numpy array - assume BGR format from OpenCV
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def detect_from_file(self, image_path: str) -> vision.HandLandmarkerResult:
        """Detect hand landmarks from an image file.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            HandLandmarkerResult containing detected hand landmarks and handedness.
        """
        return self.detect(image_path)
    
    def detect_from_array(self, image_array: np.ndarray) -> vision.HandLandmarkerResult:
        """Detect hand landmarks from a numpy array.
        
        Args:
            image_array: Image as numpy array (BGR format from OpenCV)
        
        Returns:
            HandLandmarkerResult containing detected hand landmarks and handedness.
        """
        return self.detect(image_array)


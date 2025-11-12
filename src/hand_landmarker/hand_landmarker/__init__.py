"""
Hand Landmarker Package

A Python package for detecting hand landmarks using MediaPipe.
"""

from .core.detector import HandLandmarkDetector
from .core.config import DetectorConfig
from .utils.visualization import draw_landmarks_on_image

__version__ = "0.1.0"
__all__ = [
    "HandLandmarkDetector",
    "DetectorConfig",
    "draw_landmarks_on_image",
]


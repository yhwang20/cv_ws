"""Configuration classes for hand landmark detection."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectorConfig:
    """Configuration for HandLandmarkDetector.
    
    Attributes:
        model_path: Path to the MediaPipe hand landmarker model file (.task)
        num_hands: Maximum number of hands to detect (default: 2)
        min_hand_detection_confidence: Minimum confidence for hand detection (0.0-1.0)
        min_hand_presence_confidence: Minimum confidence for hand presence (0.0-1.0)
        min_tracking_confidence: Minimum confidence for tracking (0.0-1.0)
    """
    model_path: str
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


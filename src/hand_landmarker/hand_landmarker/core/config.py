from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectorConfig:
    model_path: str
    num_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

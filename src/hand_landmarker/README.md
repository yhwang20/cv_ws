# Hand Landmarker Package

A Python package for detecting hand landmarks in images using MediaPipe. This package provides a clean, easy-to-use interface for hand landmark detection that can be extended for real-time processing.

## Installation

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the MediaPipe hand landmarker model:
```bash
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

4. Install the package (optional, for development):
```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
import cv2
from hand_landmarker import HandLandmarkDetector, draw_landmarks_on_image

# Initialize detector
detector = HandLandmarkDetector("hand_landmarker.task")

# Load image
image = cv2.imread("image.jpg")

# Detect hand landmarks
result = detector.detect(image)

# Visualize results
annotated_image = draw_landmarks_on_image(
    cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    result
)

# Display (convert back to BGR for OpenCV)
cv2.imshow("Hand Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Using Configuration

```python
from hand_landmarker import HandLandmarkDetector, DetectorConfig

# Create custom configuration
config = DetectorConfig(
    model_path="hand_landmarker.task",
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)

detector = HandLandmarkDetector(config)
```

## Examples

See the `examples/` directory for complete working examples:
- `basic_detection.py` - Simple image detection example

## Package Structure

```
hand_landmarker/
├── core/
│   ├── detector.py      # Main HandLandmarkDetector class
│   └── config.py        # Configuration classes
├── utils/
│   └── visualization.py # Drawing and visualization utilities
└── __init__.py
```

## License

This package uses MediaPipe, which is licensed under the Apache License 2.0.


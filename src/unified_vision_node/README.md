# cv_ws
cv 2025 course project, implementing hand tracking for robot control

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/mariana-smith/cv_ws.git
cd cv_ws

# Install Python dependencies for hand_landmarker (using --user flag to avoid system conflicts)
pip install --user -r src/hand_landmarker/requirements.txt

# Download the MediaPipe hand landmarker model
cd src/hand_landmarker
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
cd ../..

# Build the ROS2 workspace
colcon build

# Source the workspace
source setup.bash

# Launch 
ros2 launch unified_vision_node unified_vision.launch.py
```


## Model Selection

Choose your pinch and pose detection models:

```bash
# MediaPipe gesture recognition + MediaPipe pose (default, baseline)
ros2 launch unified_vision_node unified_vision.launch.py pinch:=mediapipegesture pose:=yolo

# MediaPipe finger pinch + YOLO pose
ros2 launch unified_vision_node unified_vision.launch.py pinch:=mediapipefinger pose:=mediapipe
```

**Available pinch models:** `custom`, `mediapipefinger`, `mediapipegesture`  
**Available pose models:** `mediapipe`, `yolo`

## Pinch Models
Custom: A custom CNN for hand pinch classification, fine tuned on our custom dataset.

MediaPipeFinger: Takes Mediapipe hand landmarks (fingertips) and computes if the hand is pinching.

MediaPipeGesture: MediaPipe Gesture Recognition model out-of-the-box (can classify open-palm, closed-fist, etc.)

## Pose Models
MediaPipe: MediaPipe pose detection for wrist and shoulder xyz

YOLO: YOLO11 pose detection for wrist and shoulder xyz



## Viewing Output

To echo the pinch and hand position topics:
```bash
# Pinch state (std_msgs/Bool)
ros2 topic echo /is_pinching

# Wrist position (geometry_msgs/PoseStamped)
ros2 topic echo /wrist_position

# Shoulder position (geometry_msgs/PoseStamped)
ros2 topic echo /shoulder_position
```


## Package Structure

```
unified_vision_node/
├── config/              # RViz configuration
├── launch/              # Launch files
├── weights/             # Model weights (installed with package)
│   ├── hand_landmarker.task      # MediaPipe hand model (~8MB)
│   ├── hand_gesture_model.pth     # Custom PyTorch CNN (~19MB)
│   ├── gesture_recognizer.task    # MediaPipe gesture recognizer (~9MB)
│   ├── yolo11n-pose.pt            # YOLO11 pose model (~5MB)
│   └── best_model_checkpoint.pth  # Alternative CNN weights
├── unified_vision_node/
│   ├── models/          # Model wrapper classes
│   │   ├── custom_finger_pinch.py           # CNN-based pinch detection
│   │   ├── mediapipe_finger_pinch.py        # Geometric pinch detection
│   │   ├── mediapipe_gesture_recognition.py # Gesture-based pinch detection
│   │   ├── mediapipe_shoulder_wrist.py      # MediaPipe body pose tracking
│   │   └── yolo_shoulder_wrist.py           # YOLO body pose tracking
│   └── nodes/
│       └── unified_vision_node.py            # Main ROS2 node
├── package.xml
├── setup.py
├── requirements.txt
└── README.md
```

## System Dependencies

**ROS2 packages:**
- ROS2 Humble or later
- `realsense2_camera` (for Intel RealSense camera)
- `rviz2`
- `tf2_ros`

**Python packages:**
See `requirements.txt` for full list. Install with:
```bash
pip install --user -r src/unified_vision_node/requirements.txt
```


















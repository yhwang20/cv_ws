# Unified Vision Node

A self-contained ROS2 package for real-time hand and body tracking with pinch detection for robot control.

## Features

- **Hand Pinch Detection**: Choose between:
  - MediaPipe-based geometric algorithm (Mariana's algorithm)
  - Custom CNN-based classifier (Owen's model)
- **Body Pose Tracking**: MediaPipe Pose for shoulder and wrist 3D localization
- **Unified Interface**: Single node combining all vision capabilities
- **Self-Contained**: All models and weights included in package

## Package Structure

```
unified_vision_node/
├── config/              # RViz configuration
├── launch/              # Launch files
├── weights/             # Model weights (installed with package)
│   ├── hand_landmarker.task      # MediaPipe hand model (~8MB)
│   ├── hand_gesture_model.pth     # Custom PyTorch CNN (~19MB)
│   └── best_model_checkpoint.pth  # Alternative CNN weights
├── unified_vision_node/
│   ├── models/          # Model wrapper classes
│   │   ├── custom_finger_pinch.py       # CNN-based pinch detection
│   │   ├── mediapipe_finger_pinch.py    # Geometric pinch detection
│   │   └── mediapipe_shoulder_wrist.py  # Body pose tracking
│   └── nodes/
│       └── unified_vision_node.py        # Main ROS2 node
├── package.xml
├── setup.py
├── requirements.txt
└── README.md
```

## Dependencies

### System Dependencies (ROS2)
- ROS2 Humble or later
- `realsense2_camera` (for Intel RealSense camera)
- `rviz2`
- `tf2_ros`

### Python Dependencies
Install with pip:
```bash
pip install -r requirements.txt
```

Required packages:
- `mediapipe>=0.10.0` - Hand and pose detection
- `opencv-python>=4.5.0` - Image processing
- `numpy>=1.21.0` - Numerical operations
- `torch>=2.0.0` - PyTorch for custom CNN
- `torchvision>=0.15.0` - PyTorch vision utilities

## Installation

1. **Clone and navigate to workspace:**
```bash
cd ~/cv_ws/src
```

2. **Install Python dependencies:**
```bash
pip install --user -r unified_vision_node/requirements.txt
```

3. **Build the package:**
```bash
cd ~/cv_ws
colcon build --packages-select unified_vision_node --symlink-install
source install/setup.bash
```

## Usage

### Basic Launch

Start the complete vision system with default settings:
```bash
ros2 launch unified_vision_node unified_vision.launch.py
```

This will launch:
- RealSense camera node
- Unified vision processing node
- RViz visualization
- TF static transforms

### Custom Launch Parameters

Override default parameters:
```bash
ros2 launch unified_vision_node unified_vision.launch.py \
    hand_side:=LEFT \
    pinch_threshold:=0.5 \
    depth_window_size:=10 \
    pose_model_complexity:=1
```

### Available Launch Arguments

**Model Paths:**
- `hand_landmarker_model_path` - Path to hand model (default: `hand_landmarker.task`)
- `custom_pinch_model_path` - Path to CNN model (default: `hand_gesture_model.pth`)

**Hand Detection:**
- `min_hand_detection_confidence` - Confidence threshold (default: 0.5)
- `min_hand_tracking_confidence` - Tracking threshold (default: 0.5)
- `pinch_threshold` - Pinch sensitivity, lower = more sensitive (default: 0.65)

**Pose Detection:**
- `min_pose_detection_confidence` - Confidence threshold (default: 0.5)
- `min_pose_tracking_confidence` - Tracking threshold (default: 0.5)
- `pose_model_complexity` - Model complexity: 0, 1, or 2 (default: 2)
- `hand_side` - Which hand to track: RIGHT or LEFT (default: RIGHT)
- `depth_window_size` - Rolling average window for depth smoothing (default: 20)

## Topics

### Subscribed Topics
- `/camera/camera/color/image_raw` (sensor_msgs/Image) - RGB camera feed
- `/camera/camera/depth/image_rect_raw` (sensor_msgs/Image) - Depth camera feed
- `/camera/camera/color/camera_info` (sensor_msgs/CameraInfo) - Camera intrinsics

### Published Topics
- `/image_annotated` (sensor_msgs/Image) - Annotated video with visualizations
- `/camera_info` (sensor_msgs/CameraInfo) - Camera info republished
- `/is_pinching` (std_msgs/Bool) - Pinch state (True/False)
- `/wrist_in_camera` (geometry_msgs/PoseStamped) - Wrist 3D position
- `/shoulder_in_camera` (geometry_msgs/PoseStamped) - Shoulder 3D position
- `/wrist_in_shoulder` (geometry_msgs/PoseStamped) - Wrist relative to shoulder

### TF Frames
- `wrist_position` - Wrist position in camera_link frame
- `shoulder_position` - Shoulder position in camera_link frame

## Switching Between Pinch Detection Models

The node is currently configured to use the **custom CNN model**. To switch between models:

### Option 1: Edit the node code (lines 66-77 in unified_vision_node.py)

**For MediaPipe geometric detection:**
```python
self.mediapipe_finger_pinch = MediapipeFingerPinch(
    model_path=hand_model_path,
    min_detection_confidence=min_hand_detection_conf,
    min_tracking_confidence=min_hand_tracking_conf,
    pinch_threshold=pinch_threshold
)
```

**For custom CNN model:**
```python
self.mediapipe_finger_pinch = CustomFingerPinch(
    model_path=custom_pinch_model_path,
    threshold=0.5
)
```

### Option 2: Make it a launch parameter (future improvement)

Add a `pinch_model_type` parameter to dynamically choose at runtime.

## Model Details

### Hand Landmarker (MediaPipe)
- **File**: `hand_landmarker.task` (7.5 MB)
- **Purpose**: Detects 21 hand landmarks for geometric pinch detection
- **Source**: MediaPipe pre-trained model

### Custom CNN (PyTorch)
- **File**: `hand_gesture_model.pth` (19 MB)
- **Architecture**: EfficientNet-B0 based binary classifier
- **Output**: Pinch probability with temporal smoothing
- **Training**: Custom dataset by Owen

### Pose Detection (MediaPipe)
- **Built-in**: No external weights needed
- **Purpose**: Full body pose with 33 landmarks
- **Used for**: Shoulder and wrist 3D localization

## Performance Tuning

### Reduce Latency
- Use `pose_model_complexity:=0` (faster but less accurate)
- Decrease `depth_window_size` (more responsive but noisier)

### Improve Stability
- Increase `depth_window_size:=30` (smoother depth)
- Increase confidence thresholds

### Adjust Sensitivity
- `pinch_threshold` - Lower values = easier to trigger pinch
- For CNN model: Adjust `threshold` parameter (default: 0.5)

## Troubleshooting

### Model files not found
The package automatically searches for models in:
1. Absolute paths (if provided)
2. `share/unified_vision_node/weights/` (after installation)
3. `src/unified_vision_node/weights/` (in source workspace)

### Camera not detected
Ensure RealSense camera is connected and drivers installed:
```bash
rs-enumerate-devices
```

### Build errors
Clean and rebuild:
```bash
cd ~/cv_ws
rm -rf build/ install/ log/
colcon build --packages-select unified_vision_node
```

## Architecture Notes

This package consolidates functionality previously split across:
- `hand_landmarker` - Now integrated as `MediapipeFingerPinch` model
- `hand_body_detection` - Now integrated as `MediapipeShoulderWrist` model  
- `pinch_detection` - Weights moved to `weights/` directory

All dependencies are now self-contained within this single package.

## Credits

- **Mariana**: Geometric pinch detection algorithm, package integration
- **Owen**: Custom CNN pinch detection model
- **Yusoo**: Body pose tracking implementation
- **MediaPipe**: Hand and pose detection frameworks

## License

Apache-2.0

## Future Improvements

- [ ] Dynamic model switching via launch parameter
- [ ] Support for multiple hand tracking
- [ ] Gesture recognition beyond pinch
- [ ] Performance benchmarking tools
- [ ] Docker container support


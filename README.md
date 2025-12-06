# cv_ws
cv 2025 course project, implementing hand tracking for robot control

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/mariana-smith/cv_ws.git
cd cv_ws

# Install Python dependencies (using --user flag to avoid system conflicts)
pip install --user -r src/unified_vision_node/requirements.txt

# Build the ROS2 workspace
colcon build

# Source the workspace
source setup.bash

# Launch the unified vision system
ros2 launch unified_vision_node unified_vision.launch.py
```

## Running (after initial setup)

For subsequent sessions, source the workspace:
```bash
cd cv_ws
source setup.bash
ros2 launch unified_vision_node unified_vision.launch.py
```

## Model Selection

Choose your pinch and pose detection models:
```bash
# Custom CNN pinch + YOLO pose (default, fastest)
ros2 launch unified_vision_node unified_vision.launch.py pinch:=custom pose:=yolo

# MediaPipe finger pinch + MediaPipe pose
ros2 launch unified_vision_node unified_vision.launch.py pinch:=mediapipefinger pose:=mediapipe

# MediaPipe gesture recognition + YOLO pose
ros2 launch unified_vision_node unified_vision.launch.py pinch:=mediapipegesture pose:=yolo
```

**Available pinch models:** `custom`, `mediapipefinger`, `mediapipegesture`  
**Available pose models:** `mediapipe`, `yolo`

## Viewing Output

To view annotated image stream:
```bash
ros2 run rqt_image_view rqt_image_view
```
Topic: `/image_annotated`

To echo the pinch and pose topics:
```bash
# Pinch state (std_msgs/Bool)
ros2 topic echo /is_pinching

# Wrist position (geometry_msgs/PoseStamped)
ros2 topic echo /wrist_position

# Shoulder position (geometry_msgs/PoseStamped)
ros2 topic echo /shoulder_position
```
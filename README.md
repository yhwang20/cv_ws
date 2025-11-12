# cv_ws
cv 2025 course project, implementing hand tracking for robot control

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/mariana-smith/cv_ws.git
cd cv_ws

# Install Python dependencies for hand_landmarker
pip install -r src/hand_landmarker/requirements.txt

# Download the MediaPipe hand landmarker model
cd src/hand_landmarker
wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
cd ../..

# Build the ROS2 workspace
colcon build

# Source the workspace
source install/setup.bash

# Launch the hand landmarker node
ros2 launch hand_landmarker hand_landmarker.launch.py
```

## Viewing Output

To view image stream:
```bash
ros2 run rqt_image_view rqt_image_view
```
Topics: `/image`, `/image_annotated`

To view pinch and hand position (pixels):
```bash
ros2 topic echo /is_pinching
ros2 topic echo /hand_pos
```
```
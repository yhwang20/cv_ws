# Hand and Body Detection Package

A ROS2 package for real-time hand and body pose detection using MediaPipe Pose.

## Features

- Real-time pose detection from camera feed
- Publishes wrist and shoulder positions as PoseStamped messages
- Broadcasts TF frames for wrist and shoulder positions
- Rolling average depth smoothing for stable 3D position estimates
- Publishes annotated video with visualization
- Standalone test scripts for image and video processing

## Dependencies

- ROS2 (Humble or later)
- MediaPipe
- OpenCV
- NumPy

## Installation

```bash
cd ~/cv_ws
colcon build --packages-select hand_body_detection
source install/setup.bash
```

## Usage

### Launch the pose detector node

```bash
ros2 launch hand_body_detection pose_detector.launch.py
```

### Launch with custom parameters

```bash
ros2 launch hand_body_detection pose_detector.launch.py hand_side:=LEFT min_detection_confidence:=0.7 depth_window_size:=10
```

### Run standalone test scripts

Test on images:
```bash
ros2 run hand_body_detection pose_image_test
```

Test on video feed:
```bash
ros2 run hand_body_detection pose_video_test
```

## Topics

### Subscribed Topics

- `/camera/camera/color/image_raw` (sensor_msgs/Image): RGB camera feed
- `/camera/camera/depth/image_rect_raw` (sensor_msgs/Image): Depth camera feed
- `/camera/camera/color/camera_info` (sensor_msgs/CameraInfo): Camera intrinsics

### Published Topics

- `/wrist_position` (geometry_msgs/PoseStamped): 3D position of the wrist
- `/shoulder_position` (geometry_msgs/PoseStamped): 3D position of the shoulder
- `/image_annotated_body` (sensor_msgs/Image): Annotated video feed with pose overlay

### TF Frames

- `wrist_position`: Wrist position relative to `camera_link`
- `shoulder_position`: Shoulder position relative to `camera_link`

## Parameters

- `min_detection_confidence` (double, default: 0.5): Minimum confidence for pose detection
- `min_tracking_confidence` (double, default: 0.5): Minimum confidence for pose tracking
- `model_complexity` (int, default: 2): Pose model complexity (0, 1, or 2)
- `hand_side` (string, default: 'RIGHT'): Which hand to track ('RIGHT' or 'LEFT')
- `depth_window_size` (int, default: 5): Rolling average window size for depth smoothing

## Notes

### Depth Smoothing

The node implements rolling average smoothing on depth measurements to reduce jitter and noise in the 3D position estimates. The `depth_window_size` parameter controls how many recent depth measurements are averaged together:

- **Smaller values (1-3)**: More responsive to movement, but noisier
- **Medium values (5-10)**: Good balance between responsiveness and stability (recommended)
- **Larger values (10+)**: Very smooth, but slower to respond to rapid movements

The smoothing is applied independently to wrist and shoulder depth measurements and affects both the PoseStamped messages and TF broadcasts.

## License

Apache-2.0


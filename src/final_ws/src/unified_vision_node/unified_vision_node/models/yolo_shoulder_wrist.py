import numpy as np
import cv2
from collections import deque
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory

# YOLO11 pose detection for wrist and shoulder xyz


class YoloShoulderWrist:
    """
    YOLO-based shoulder and wrist detection with 3D localization.
    
    YOLO11 Pose Keypoint Indices:
    0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
    5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
    9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
    13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
    """
    
    def __init__(self, 
                 model_path='yolo11n-pose.pt',
                 hand_side='RIGHT',
                 depth_window_size=20,
                 confidence_threshold=0.5):
        """
        Initialize YOLO pose detector.
        
        Args:
            model_path: Path to YOLO pose model weights
            hand_side: Which hand to track ('RIGHT' or 'LEFT')
            depth_window_size: Size of rolling window for depth smoothing
            confidence_threshold: Minimum confidence for detections
        """
        self.hand_side = hand_side
        self.confidence_threshold = confidence_threshold
        
        # Resolve model path - try as absolute first, then in package weights directory
        if not os.path.isabs(model_path) or not os.path.exists(model_path):
            try:
                package_share_dir = get_package_share_directory('unified_vision_node')
                weights_path = os.path.join(package_share_dir, 'weights', model_path)
                if os.path.exists(weights_path):
                    model_path = weights_path
            except Exception:
                # Fall back to relative path from this file
                weights_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'weights')
                weights_path = os.path.join(weights_dir, model_path)
                if os.path.exists(weights_path):
                    model_path = weights_path
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Depth buffers for temporal smoothing
        self.wrist_depth_buffer = deque(maxlen=depth_window_size)
        self.shoulder_depth_buffer = deque(maxlen=depth_window_size)
        
        # Define keypoint indices based on hand side
        if self.hand_side.upper() == 'RIGHT':
            self.shoulder_idx = 6  # Right shoulder
            self.wrist_idx = 10    # Right wrist
            self.elbow_idx = 8     # Right elbow
        else:
            self.shoulder_idx = 5  # Left shoulder
            self.wrist_idx = 9     # Left wrist
            self.elbow_idx = 7     # Left elbow
    
    def __call__(self, rgb_image, depth_image, camera_info):
        """
        Process image and detect shoulder and wrist positions.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            depth_image: Depth image as numpy array (H, W) in millimeters
            camera_info: Dictionary with camera intrinsics (fx, fy, cx, cy)
        
        Returns:
            Dictionary containing:
                - 'wrist_position': (x, y, z) in meters or None
                - 'shoulder_position': (x, y, z) in meters or None
                - 'wrist_pixel': (x, y) pixel coordinates or None
                - 'shoulder_pixel': (x, y) pixel coordinates or None
                - 'pose_landmarks': raw YOLO results
                - 'annotated_image': RGB image with visualizations
        """
        # Run YOLO inference
        results = self.model(rgb_image, verbose=False, conf=self.confidence_threshold)
        
        # Initialize output
        output = {
            'wrist_position': None,
            'shoulder_position': None,
            'wrist_pixel': None,
            'shoulder_pixel': None,
            'pose_landmarks': results,
            'annotated_image': rgb_image.copy()
        }
        
        # Process results (only use first detection)
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints
            
            # Check if keypoints were detected
            if keypoints.xy is not None and len(keypoints.xy) > 0:
                kp_xy = keypoints.xy[0]  # Get first person's keypoints (shape: [17, 2])
                
                # Extract shoulder and wrist positions
                shoulder_pixel = self._extract_keypoint(kp_xy, self.shoulder_idx)
                wrist_pixel = self._extract_keypoint(kp_xy, self.wrist_idx)
                
                # Draw YOLO skeleton
                output['annotated_image'] = results[0].plot()
                
                # Add custom markers for shoulder and wrist
                if shoulder_pixel is not None:
                    output['shoulder_pixel'] = shoulder_pixel
                    cv2.circle(output['annotated_image'], shoulder_pixel, 12, (0, 255, 0), -1)
                    cv2.circle(output['annotated_image'], shoulder_pixel, 10, (255, 255, 255), 2)
                    cv2.putText(output['annotated_image'], 'SHOULDER', 
                               (shoulder_pixel[0] + 15, shoulder_pixel[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                if wrist_pixel is not None:
                    output['wrist_pixel'] = wrist_pixel
                    cv2.circle(output['annotated_image'], wrist_pixel, 12, (255, 0, 0), -1)
                    cv2.circle(output['annotated_image'], wrist_pixel, 10, (255, 255, 255), 2)
                    cv2.putText(output['annotated_image'], 'WRIST',
                               (wrist_pixel[0] + 15, wrist_pixel[1]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Calculate 3D positions with depth
                if depth_image is not None and camera_info is not None:
                    if wrist_pixel is not None:
                        wrist_depth_raw = self._get_depth_at_pixel(
                            depth_image, wrist_pixel[0], wrist_pixel[1]
                        )
                        if wrist_depth_raw is not None:
                            self.wrist_depth_buffer.append(wrist_depth_raw)
                    
                    if shoulder_pixel is not None:
                        shoulder_depth_raw = self._get_depth_at_pixel(
                            depth_image, shoulder_pixel[0], shoulder_pixel[1]
                        )
                        if shoulder_depth_raw is not None:
                            self.shoulder_depth_buffer.append(shoulder_depth_raw)
                    
                    # Get averaged depth values
                    wrist_depth_avg = self._get_averaged_depth(self.wrist_depth_buffer)
                    shoulder_depth_avg = self._get_averaged_depth(self.shoulder_depth_buffer)
                    
                    # Convert to 3D coordinates
                    if wrist_pixel is not None and wrist_depth_avg is not None:
                        output['wrist_position'] = self._pixel_to_3d(
                            wrist_pixel[0], wrist_pixel[1], wrist_depth_avg, camera_info
                        )
                    
                    if shoulder_pixel is not None and shoulder_depth_avg is not None:
                        output['shoulder_position'] = self._pixel_to_3d(
                            shoulder_pixel[0], shoulder_pixel[1], shoulder_depth_avg, camera_info
                        )
        
        return output
    
    def _extract_keypoint(self, keypoints, index):
        """
        Extract keypoint coordinates and validate.
        
        Args:
            keypoints: YOLO keypoints array (shape: [17, 2])
            index: Keypoint index
        
        Returns:
            Tuple (x, y) as integers or None if invalid
        """
        if index >= len(keypoints):
            return None
        
        x, y = keypoints[index]
        
        # Check if keypoint is valid (YOLO uses 0,0 for missing keypoints)
        if x == 0 and y == 0:
            return None
        
        return (int(x), int(y))
    
    def _get_depth_at_pixel(self, depth_image, pixel_x, pixel_y):
        """
        Get depth value at specific pixel coordinates.
        
        Args:
            depth_image: Depth image (H, W) in millimeters (uint16)
            pixel_x: X coordinate
            pixel_y: Y coordinate
        
        Returns:
            Depth in meters or None if invalid
        """
        if depth_image is None:
            return None
        
        depth_height, depth_width = depth_image.shape[:2]
        
        # Check bounds
        if pixel_x < 0 or pixel_x >= depth_width or pixel_y < 0 or pixel_y >= depth_height:
            return None
        
        depth_value = depth_image[pixel_y, pixel_x]
        
        # Convert to meters (RealSense depth is in millimeters as uint16)
        if depth_image.dtype == np.uint16:
            depth_meters = float(depth_value) / 1000.0
        else:
            depth_meters = float(depth_value)
        
        # Validate depth
        if depth_meters <= 0 or np.isnan(depth_meters):
            return None
        
        return depth_meters
    
    def _get_averaged_depth(self, depth_buffer):
        """
        Calculate average depth from buffer for temporal smoothing.
        
        Args:
            depth_buffer: Deque containing recent depth measurements
        
        Returns:
            Average depth or None if buffer is empty
        """
        if len(depth_buffer) == 0:
            return None
        
        return np.mean(depth_buffer)
    
    def _pixel_to_3d(self, pixel_x, pixel_y, depth_value, camera_info):
        """
        Convert 2D pixel coordinates + depth to 3D coordinates.
        
        Uses pinhole camera model:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth
        
        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            depth_value: Depth in meters
            camera_info: Dictionary with fx, fy, cx, cy
        
        Returns:
            Tuple (x, y, z) in meters or None if invalid
        """
        if camera_info is None:
            return None
        
        if depth_value <= 0 or np.isnan(depth_value):
            return None
        
        # Apply pinhole camera model
        x = (pixel_x - camera_info['cx']) * depth_value / camera_info['fx']
        y = (pixel_y - camera_info['cy']) * depth_value / camera_info['fy']
        z = depth_value
        
        return (x, y, z)

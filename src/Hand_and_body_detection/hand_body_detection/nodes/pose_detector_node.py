#!/usr/bin/env python3
"""
ROS 2 Node for real-time body pose detection.

Subscribes to /camera/camera/color/image_raw and publishes:
- /wrist_position (geometry_msgs/PoseStamped)
- /shoulder_position (geometry_msgs/PoseStamped)
- /image_annotated_body (sensor_msgs/Image)
- TF frames: wrist_position and shoulder_position

"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from collections import deque
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pose_detector import PoseDetector


class PoseDetectorNode(Node):
    """ROS 2 node for body pose detection."""
    
    def __init__(self):
        """Initialize the pose detector node."""
        super().__init__('pose_detector')
        
        # Declare parameters
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('model_complexity', 2)
        self.declare_parameter('hand_side', 'RIGHT')  # 'RIGHT' or 'LEFT'
        self.declare_parameter('depth_window_size', 20)  # Rolling average window size
        
        # Get parameters
        min_detection_confidence = self.get_parameter('min_detection_confidence').get_parameter_value().double_value
        min_tracking_confidence = self.get_parameter('min_tracking_confidence').get_parameter_value().double_value
        model_complexity = self.get_parameter('model_complexity').get_parameter_value().integer_value
        self.hand_side = self.get_parameter('hand_side').get_parameter_value().string_value
        depth_window_size = self.get_parameter('depth_window_size').get_parameter_value().integer_value
        
        # Initialize cv_bridge
        self.bridge = CvBridge()
        
        # Store latest depth image and camera intrinsics
        self.latest_depth_image = None
        self.camera_intrinsics = None
        
        # Rolling average buffers for depth smoothing
        self.wrist_depth_buffer = deque(maxlen=depth_window_size)
        self.shoulder_depth_buffer = deque(maxlen=depth_window_size)
        
        # Initialize pose detector
        try:
            self.detector = PoseDetector(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity
            )
            self.get_logger().info('Pose detector initialized')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize detector: {e}')
            raise
        
        # Create subscriber to color image topic
        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        # Create subscriber to depth image topic
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )
        
        # Create subscriber to camera info topic for intrinsics
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Create publisher for annotated images
        self.image_pub = self.create_publisher(Image, '/image_annotated_body', 10)
        
        # Create publishers for wrist and shoulder positions
        self.wrist_pos_pub = self.create_publisher(PoseStamped, '/wrist_position', 10)
        self.shoulder_pos_pub = self.create_publisher(PoseStamped, '/shoulder_position', 10)
        
        # Create TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info('Pose detector node started')
        self.get_logger().info(f'Tracking {self.hand_side} hand')
        self.get_logger().info(f'Depth smoothing enabled: rolling average window size = {depth_window_size}')
        self.get_logger().info('Subscribing to: /camera/camera/color/image_raw, /camera/camera/depth/image_rect_raw')
        self.get_logger().info('Publishing to: /image_annotated_body, /wrist_position, /shoulder_position')
        self.get_logger().info('Broadcasting TF: wrist_position, shoulder_position')
    
    def depth_callback(self, msg):
        """Store the latest depth image."""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')
    
    def camera_info_callback(self, msg):
        """Store camera intrinsics from CameraInfo message."""
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
                'frame_id': msg.header.frame_id
            }
            self.get_logger().info(
                f'Camera intrinsics received: fx={self.camera_intrinsics["fx"]:.2f}, '
                f'fy={self.camera_intrinsics["fy"]:.2f}, '
                f'cx={self.camera_intrinsics["cx"]:.2f}, '
                f'cy={self.camera_intrinsics["cy"]:.2f}'
            )
    
    def get_depth_at_pixel(self, pixel_x, pixel_y):
        """
        Get depth value at pixel location from depth image.
        
        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            
        Returns:
            float: Depth value in meters, or None if invalid
        """
        if self.latest_depth_image is None:
            return None
        
        depth_height, depth_width = self.latest_depth_image.shape[:2]
        
        # Ensure pixel is within bounds
        if pixel_x < 0 or pixel_x >= depth_width or pixel_y < 0 or pixel_y >= depth_height:
            return None
        
        # Read depth value
        depth_value = self.latest_depth_image[pixel_y, pixel_x]
        
        # Convert to meters (RealSense depth is uint16 in mm)
        if self.latest_depth_image.dtype == np.uint16:
            depth_meters = float(depth_value) / 1000.0
        else:
            depth_meters = float(depth_value)
        
        # Return None if depth is invalid
        if depth_meters <= 0 or np.isnan(depth_meters):
            return None
        
        return depth_meters
    
    def pixel_to_3d(self, pixel_x, pixel_y, depth_value):
        """
        Convert pixel coordinates + depth to 3D world coordinates.
        
        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            depth_value: Depth value in meters
            
        Returns:
            tuple: (x, y, z) in meters in camera frame, or None if conversion fails
        """
        if self.camera_intrinsics is None:
            return None
        
        if depth_value <= 0 or np.isnan(depth_value):
            return None
        
        # Convert using pinhole camera model
        x = (pixel_x - self.camera_intrinsics['cx']) * depth_value / self.camera_intrinsics['fx']
        y = (pixel_y - self.camera_intrinsics['cy']) * depth_value / self.camera_intrinsics['fy']
        z = depth_value
        
        return (x, y, z)
    
    def get_averaged_depth(self, depth_buffer):
        """
        Get the average depth from the rolling window buffer.
        
        Args:
            depth_buffer: deque containing recent depth measurements
            
        Returns:
            float: Averaged depth value, or None if buffer is empty
        """
        if len(depth_buffer) == 0:
            return None
        
        return np.mean(depth_buffer)
    
    def publish_pose_and_tf(self, name, coords_3d, timestamp, frame_id='camera_link'):
        """
        Publish PoseStamped message and TF transform for a landmark.
        
        Args:
            name: Name of the landmark (e.g., 'wrist', 'shoulder')
            coords_3d: (x, y, z) coordinates in meters
            timestamp: ROS timestamp
            frame_id: Frame ID for the transform
        """
        # Publish PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = frame_id
        
        pose_msg.pose.position.x = coords_3d[0]
        pose_msg.pose.position.y = coords_3d[1]
        pose_msg.pose.position.z = coords_3d[2]
        
        # Default orientation (identity quaternion)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        
        if name == 'wrist':
            self.wrist_pos_pub.publish(pose_msg)
        elif name == 'shoulder':
            self.shoulder_pos_pub.publish(pose_msg)
        
        # Broadcast TF
        tf_msg = TransformStamped()
        tf_msg.header.stamp = timestamp
        tf_msg.header.frame_id = frame_id
        tf_msg.child_frame_id = f'{name}_position'
        
        tf_msg.transform.translation.x = coords_3d[0]
        tf_msg.transform.translation.y = coords_3d[1]
        tf_msg.transform.translation.z = coords_3d[2]
        
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = 1.0
        
        self.tf_broadcaster.sendTransform(tf_msg)
    
    def image_callback(self, msg):
        """Process incoming image and publish results."""
        try:
            # Convert ROS Image to OpenCV (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Detect pose
            results = self.detector.detect(rgb_image)
            
            # Create annotated image
            annotated_image = rgb_image.copy()
            self.detector.draw_landmarks(annotated_image, results)
            
            # Get wrist and shoulder positions
            positions = self.detector.get_wrist_shoulder_positions(results, self.hand_side)
            
            if positions is not None:
                wrist_norm = positions['wrist']
                shoulder_norm = positions['shoulder']
                
                # Convert normalized coordinates to pixel coordinates
                height, width = rgb_image.shape[:2]
                wrist_pixel = (int(wrist_norm[0] * width), int(wrist_norm[1] * height))
                shoulder_pixel = (int(shoulder_norm[0] * width), int(shoulder_norm[1] * height))
                
                # Draw circles on wrist and shoulder
                # Wrist - Blue circle
                cv2.circle(annotated_image, wrist_pixel, 12, (255, 0, 0), -1)  # Blue filled
                cv2.circle(annotated_image, wrist_pixel, 10, (255, 255, 255), 2)  # White border
                cv2.putText(annotated_image, 'WRIST', (wrist_pixel[0] + 15, wrist_pixel[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Shoulder - Green circle
                cv2.circle(annotated_image, shoulder_pixel, 12, (0, 255, 0), -1)  # Green filled
                cv2.circle(annotated_image, shoulder_pixel, 10, (255, 255, 255), 2)  # White border
                cv2.putText(annotated_image, 'SHOULDER', (shoulder_pixel[0] + 15, shoulder_pixel[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Publish 3D positions if depth is available
                if self.latest_depth_image is not None and self.camera_intrinsics is not None:
                    # Get raw depth at wrist and shoulder
                    wrist_depth_raw = self.get_depth_at_pixel(wrist_pixel[0], wrist_pixel[1])
                    shoulder_depth_raw = self.get_depth_at_pixel(shoulder_pixel[0], shoulder_pixel[1])
                    
                    # Add to rolling average buffers
                    if wrist_depth_raw is not None:
                        self.wrist_depth_buffer.append(wrist_depth_raw)
                    
                    if shoulder_depth_raw is not None:
                        self.shoulder_depth_buffer.append(shoulder_depth_raw)
                    
                    # Get averaged depth values
                    wrist_depth_avg = self.get_averaged_depth(self.wrist_depth_buffer)
                    shoulder_depth_avg = self.get_averaged_depth(self.shoulder_depth_buffer)
                    
                    # Publish wrist position with averaged depth
                    if wrist_depth_avg is not None:
                        wrist_3d = self.pixel_to_3d(wrist_pixel[0], wrist_pixel[1], wrist_depth_avg)
                        if wrist_3d is not None:
                            self.publish_pose_and_tf('wrist', wrist_3d, msg.header.stamp)
                            self.get_logger().info(
                                f'Wrist: x={wrist_3d[0]:.3f}m, y={wrist_3d[1]:.3f}m, z={wrist_3d[2]:.3f}m (avg depth from {len(self.wrist_depth_buffer)} samples)',
                                throttle_duration_sec=1.0
                            )
                    
                    # Publish shoulder position with averaged depth
                    if shoulder_depth_avg is not None:
                        shoulder_3d = self.pixel_to_3d(shoulder_pixel[0], shoulder_pixel[1], shoulder_depth_avg)
                        if shoulder_3d is not None:
                            self.publish_pose_and_tf('shoulder', shoulder_3d, msg.header.stamp)
                            self.get_logger().info(
                                f'Shoulder: x={shoulder_3d[0]:.3f}m, y={shoulder_3d[1]:.3f}m, z={shoulder_3d[2]:.3f}m (avg depth from {len(self.shoulder_depth_buffer)} samples)',
                                throttle_duration_sec=1.0
                            )
                else:
                    if self.latest_depth_image is None:
                        self.get_logger().warn('No depth image available', throttle_duration_sec=5.0)
                    if self.camera_intrinsics is None:
                        self.get_logger().warn('No camera intrinsics available', throttle_duration_sec=5.0)
            else:
                self.get_logger().info('No pose detected', throttle_duration_sec=2.0)
            
            # Convert back to BGR for ROS
            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            # Publish annotated image
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding='bgr8')
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = PoseDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


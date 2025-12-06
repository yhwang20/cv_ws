#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from ament_index_python.packages import get_package_share_directory
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mediapipe_finger_pinch import MediapipeFingerPinch
from models.mediapipe_shoulder_wrist import MediapipeShoulderWrist
from models.yolo_shoulder_wrist import YoloShoulderWrist
from models.custom_finger_pinch import CustomFingerPinch  
from models.mediapipe_gesture_recognition import MediapipeGestureRecognition


class UnifiedVisionNode(Node):
    
    def __init__(self):
        super().__init__('unified_vision')
        
        self.declare_parameter('hand_landmarker_model_path', 'hand_landmarker.task')
        self.declare_parameter('gesture_recognizer_model_path', 'gesture_recognizer.task')
        self.declare_parameter('custom_pinch_model_path', 'hand_gesture_model.pth')
        self.declare_parameter('yolo_pose_model_path', 'yolo11n-pose.pt')
        self.declare_parameter('min_hand_detection_confidence', 0.5)
        self.declare_parameter('min_hand_tracking_confidence', 0.5)
        self.declare_parameter('pinch_threshold', 0.65)
        
        self.declare_parameter('min_pose_detection_confidence', 0.5)
        self.declare_parameter('min_pose_tracking_confidence', 0.5)
        self.declare_parameter('pose_model_complexity', 2)
        self.declare_parameter('hand_side', 'RIGHT')
        self.declare_parameter('depth_window_size', 20)
        
        # Model selection parameters
        self.declare_parameter('pinch', 'custom')
        self.declare_parameter('pose', 'mediapipe')
        
        hand_model_path = self.get_parameter('hand_landmarker_model_path').get_parameter_value().string_value
        gesture_model_path = self.get_parameter('gesture_recognizer_model_path').get_parameter_value().string_value
        custom_pinch_model_path = self.get_parameter('custom_pinch_model_path').get_parameter_value().string_value
        yolo_pose_model_path = self.get_parameter('yolo_pose_model_path').get_parameter_value().string_value
        min_hand_detection_conf = self.get_parameter('min_hand_detection_confidence').get_parameter_value().double_value
        min_hand_tracking_conf = self.get_parameter('min_hand_tracking_confidence').get_parameter_value().double_value
        pinch_threshold = self.get_parameter('pinch_threshold').get_parameter_value().double_value
        
        min_pose_detection_conf = self.get_parameter('min_pose_detection_confidence').get_parameter_value().double_value
        min_pose_tracking_conf = self.get_parameter('min_pose_tracking_confidence').get_parameter_value().double_value
        pose_complexity = self.get_parameter('pose_model_complexity').get_parameter_value().integer_value
        hand_side = self.get_parameter('hand_side').get_parameter_value().string_value
        depth_window_size = self.get_parameter('depth_window_size').get_parameter_value().integer_value
        
        # Get model selection parameters
        pinch_model = self.get_parameter('pinch').get_parameter_value().string_value
        pose_model = self.get_parameter('pose').get_parameter_value().string_value
        
        # Resolve model paths - try as absolute first, then in unified_vision_node/weights
        if not os.path.isabs(hand_model_path) or not os.path.exists(hand_model_path):
            try:
                package_share_dir = get_package_share_directory('unified_vision_node')
                full_model_path = os.path.join(package_share_dir, 'weights', hand_model_path)
                if os.path.exists(full_model_path):
                    hand_model_path = full_model_path
            except Exception:
                # Fall back to relative path from package source
                weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'weights')
                full_model_path = os.path.join(weights_dir, hand_model_path)
                if os.path.exists(full_model_path):
                    hand_model_path = full_model_path
        
        # Resolve gesture model path
        if not os.path.isabs(gesture_model_path) or not os.path.exists(gesture_model_path):
            try:
                package_share_dir = get_package_share_directory('unified_vision_node')
                full_model_path = os.path.join(package_share_dir, 'weights', gesture_model_path)
                if os.path.exists(full_model_path):
                    gesture_model_path = full_model_path
            except Exception:
                # Fall back to relative path from package source
                weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'weights')
                full_model_path = os.path.join(weights_dir, gesture_model_path)
                if os.path.exists(full_model_path):
                    gesture_model_path = full_model_path
        
        # Resolve custom pinch model path
        if not os.path.isabs(custom_pinch_model_path) or not os.path.exists(custom_pinch_model_path):
            try:
                package_share_dir = get_package_share_directory('unified_vision_node')
                full_model_path = os.path.join(package_share_dir, 'weights', custom_pinch_model_path)
                if os.path.exists(full_model_path):
                    custom_pinch_model_path = full_model_path
            except Exception:
                # Fall back to relative path from package source
                weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'weights')
                full_model_path = os.path.join(weights_dir, custom_pinch_model_path)
                if os.path.exists(full_model_path):
                    custom_pinch_model_path = full_model_path
        
        # Resolve YOLO pose model path
        if not os.path.isabs(yolo_pose_model_path) or not os.path.exists(yolo_pose_model_path):
            try:
                package_share_dir = get_package_share_directory('unified_vision_node')
                full_model_path = os.path.join(package_share_dir, 'weights', yolo_pose_model_path)
                if os.path.exists(full_model_path):
                    yolo_pose_model_path = full_model_path
            except Exception:
                # Fall back to relative path from package source
                weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'weights')
                full_model_path = os.path.join(weights_dir, yolo_pose_model_path)
                if os.path.exists(full_model_path):
                    yolo_pose_model_path = full_model_path
        
        self.bridge = CvBridge()
        
        self.latest_depth_image = None
        self.camera_intrinsics = None
        self.latest_camera_info = None
        
        try:
            # Initialize pinch detection model based on parameter
            if pinch_model == 'custom':
                self.mediapipe_finger_pinch = CustomFingerPinch(
                    model_path=custom_pinch_model_path,
                    threshold=0.5
                )
                self.get_logger().info(f'Custom CNN pinch detection initialized: {custom_pinch_model_path}')
            elif pinch_model == 'mediapipefinger':
                self.mediapipe_finger_pinch = MediapipeFingerPinch(
                    model_path=hand_model_path,
                    min_detection_confidence=min_hand_detection_conf,
                    min_tracking_confidence=min_hand_tracking_conf,
                    pinch_threshold=pinch_threshold
                )
                self.get_logger().info(f'MediaPipe finger pinch initialized: {hand_model_path}')
            elif pinch_model == 'mediapipegesture':
                self.mediapipe_finger_pinch = MediapipeGestureRecognition(
                    model_path=gesture_model_path,
                    min_detection_confidence=min_hand_detection_conf,
                    min_tracking_confidence=min_hand_tracking_conf,
                    gesture_classes=['Closed_Fist', 'Open_Palm'],
                    num_hands=2
                )
                self.get_logger().info(f'MediaPipe gesture recognition initialized: {gesture_model_path}')
            else:
                raise ValueError(f"Invalid pinch model: {pinch_model}. Must be 'custom', 'mediapipefinger', or 'mediapipegesture'")
        except Exception as e:
            self.get_logger().error(f'Failed to initialize pinch detection: {e}')
            raise
        
        try:
            # Initialize pose detection model based on parameter
            if pose_model == 'mediapipe':
                self.mediapipe_shoulder_wrist = MediapipeShoulderWrist(
                    min_detection_confidence=min_pose_detection_conf,
                    min_tracking_confidence=min_pose_tracking_conf,
                    model_complexity=pose_complexity,
                    hand_side=hand_side,
                    depth_window_size=depth_window_size
                )
                self.get_logger().info(f'MediaPipe shoulder/wrist initialized (hand: {hand_side}, window: {depth_window_size})')
            elif pose_model == 'yolo':
                self.mediapipe_shoulder_wrist = YoloShoulderWrist(
                    model_path=yolo_pose_model_path,
                    hand_side=hand_side,
                    depth_window_size=depth_window_size,
                    confidence_threshold=min_pose_detection_conf
                )
                self.get_logger().info(f'YOLO shoulder/wrist initialized: {yolo_pose_model_path} (hand: {hand_side}, window: {depth_window_size})')
            else:
                raise ValueError(f"Invalid pose model: {pose_model}. Must be 'mediapipe' or 'yolo'")
        except Exception as e:
            self.get_logger().error(f'Failed to initialize pose detection: {e}')
            raise
        
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        self.image_pub = self.create_publisher(Image, '/image_annotated', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera_info', 10)
        self.pinch_pub = self.create_publisher(Bool, '/is_pinching', 10)
        self.wrist_pos_pub = self.create_publisher(PoseStamped, '/wrist_in_camera', 10)
        self.shoulder_pos_pub = self.create_publisher(PoseStamped, '/shoulder_in_camera', 10)
        self.wrist_in_shoulder_pub = self.create_publisher(PoseStamped, '/wrist_in_shoulder', 10)
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.get_logger().info('unified vision node started')
    
    def depth_callback(self, msg):
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'error processing depth: {e}')
    
    def camera_info_callback(self, msg):
        self.latest_camera_info = msg
        
        if self.camera_intrinsics is None:
            self.camera_intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
                'frame_id': msg.header.frame_id
            }
            self.get_logger().info(f'camera intrinsics: fx={self.camera_intrinsics["fx"]:.2f}, fy={self.camera_intrinsics["fy"]:.2f}')
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            pinch_result = self.mediapipe_finger_pinch(rgb_image, self.latest_depth_image, self.camera_intrinsics)
            loc_result = self.mediapipe_shoulder_wrist(rgb_image, self.latest_depth_image, self.camera_intrinsics)
            
            # combine annotations
            combined_annotated = loc_result['annotated_image'].copy()
            hand_only = pinch_result['annotated_image']
            
            rgb_diff = cv2.absdiff(hand_only, rgb_image)
            gray_diff = cv2.cvtColor(rgb_diff, cv2.COLOR_RGB2GRAY)
            _, mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
            
            combined_annotated = np.where(mask[:, :, np.newaxis] > 0, hand_only, combined_annotated)
            
            # add status text
            hand_text = "PINCHING" if pinch_result['is_pinching'] else "OPEN"
            hand_color = (0, 255, 0) if pinch_result['is_pinching'] else (255, 0, 0)
            
            cv2.putText(combined_annotated, hand_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, hand_color, 2, cv2.LINE_AA)
            
            pinch_msg = Bool()
            pinch_msg.data = bool(pinch_result['is_pinching'])
            self.pinch_pub.publish(pinch_msg)
            
            if loc_result['wrist_position'] is not None:
                self._publish_pose_and_tf('wrist', loc_result['wrist_position'], msg.header.stamp, 'camera_link')
            
            if loc_result['shoulder_position'] is not None:
                self._publish_pose_and_tf('shoulder', loc_result['shoulder_position'], msg.header.stamp, 'camera_link')
            
            # Calculate and publish wrist position relative to shoulder
            if loc_result['wrist_position'] is not None and loc_result['shoulder_position'] is not None:
                wrist_in_shoulder = np.array(loc_result['wrist_position']) - np.array(loc_result['shoulder_position'])
                self._publish_wrist_in_shoulder(wrist_in_shoulder, msg.header.stamp)
            
            self.get_logger().info(
                f'pinch: {pinch_result["is_pinching"]} | wrist: {loc_result["wrist_position"] is not None} | shoulder: {loc_result["shoulder_position"] is not None}',
                throttle_duration_sec=1.0
            )
            
            annotated_bgr = cv2.cvtColor(combined_annotated, cv2.COLOR_RGB2BGR)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding='bgr8')
            annotated_msg.header = msg.header
            self.image_pub.publish(annotated_msg)
            
            if self.latest_camera_info is not None:
                camera_info_msg = CameraInfo()
                camera_info_msg.header.stamp = msg.header.stamp
                camera_info_msg.header.frame_id = self.latest_camera_info.header.frame_id
                camera_info_msg.height = self.latest_camera_info.height
                camera_info_msg.width = self.latest_camera_info.width
                camera_info_msg.distortion_model = self.latest_camera_info.distortion_model
                camera_info_msg.d = self.latest_camera_info.d
                camera_info_msg.k = self.latest_camera_info.k
                camera_info_msg.r = self.latest_camera_info.r
                camera_info_msg.p = self.latest_camera_info.p
                camera_info_msg.binning_x = self.latest_camera_info.binning_x
                camera_info_msg.binning_y = self.latest_camera_info.binning_y
                camera_info_msg.roi = self.latest_camera_info.roi
                
                self.camera_info_pub.publish(camera_info_msg)
            
        except Exception as e:
            self.get_logger().error(f'error processing image: {e}')
    
    def _publish_pose_and_tf(self, name, coords_3d, timestamp, frame_id='camera_link'):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = frame_id
        
        pose_msg.pose.position.x = coords_3d[0]
        pose_msg.pose.position.y = coords_3d[1]
        pose_msg.pose.position.z = coords_3d[2]
        
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        
        if name == 'wrist':
            self.wrist_pos_pub.publish(pose_msg)
        elif name == 'shoulder':
            self.shoulder_pos_pub.publish(pose_msg)
        
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
    
    def _publish_wrist_in_shoulder(self, relative_coords, timestamp):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = 'shoulder_position'
        
        pose_msg.pose.position.x = relative_coords[0]
        pose_msg.pose.position.y = relative_coords[1]
        pose_msg.pose.position.z = relative_coords[2]
        
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0
        
        self.wrist_in_shoulder_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = UnifiedVisionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'error: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

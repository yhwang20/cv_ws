#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

from hand_landmarker import HandLandmarkDetector, draw_landmarks_on_image


class HandLandmarkerNode(Node):
    
    def __init__(self):
        super().__init__('hand_landmarker')
        
        try:
            package_share_dir = get_package_share_directory('hand_landmarker')
            default_model_path = os.path.join(package_share_dir, 'models', 'hand_landmarker.task')
        except Exception as e:
            self.get_logger().warn(f'could not get package dir: {e}')
            default_model_path = 'hand_landmarker.task'
        
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        min_detection_confidence = self.get_parameter('min_detection_confidence').get_parameter_value().double_value
        min_tracking_confidence = self.get_parameter('min_tracking_confidence').get_parameter_value().double_value
        
        self.bridge = CvBridge()
        
        self.latest_depth_image = None
        self.camera_intrinsics = None
        self.latest_camera_info = None
        
        try:
            from hand_landmarker import DetectorConfig
            
            config = DetectorConfig(
                model_path=model_path,
                min_tracking_confidence=min_tracking_confidence,
                min_hand_detection_confidence=min_detection_confidence
            )
            
            self.detector = HandLandmarkDetector(config)
            self.get_logger().info(f'hand detector initialized: {model_path}')
        except Exception as e:
            self.get_logger().error(f'failed to initialize: {e}')
            raise
        
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        
        self.image_pub = self.create_publisher(Image, '/image_annotated', 10)
        self.annotated_camera_info_pub = self.create_publisher(CameraInfo, '/camera_info', 10)
        self.pinch_pub = self.create_publisher(Bool, '/is_pinching', 10)
        
        self.declare_parameter('pinch_threshold', 0.65)
        self.pinch_threshold = self.get_parameter('pinch_threshold').get_parameter_value().double_value
        
        self.declare_parameter('debug_depth', False)
        self.debug_depth = self.get_parameter('debug_depth').get_parameter_value().bool_value
        
        self.get_logger().info('hand landmarker node started')
    
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
    
    def get_depth_at_pixel(self, pixel_x, pixel_y):
        if self.latest_depth_image is None:
            return None
        
        depth_height, depth_width = self.latest_depth_image.shape[:2]
        
        if pixel_x < 0 or pixel_x >= depth_width or pixel_y < 0 or pixel_y >= depth_height:
            return None
        
        depth_value = self.latest_depth_image[pixel_y, pixel_x]
        
        if self.latest_depth_image.dtype == np.uint16:
            depth_meters = float(depth_value) / 1000.0
        else:
            depth_meters = float(depth_value)
        
        if depth_meters <= 0 or np.isnan(depth_meters):
            return None
        
        return depth_meters
    
    def pixel_to_3d(self, pixel_x, pixel_y, depth_value):
        if self.camera_intrinsics is None:
            return None
        
        if depth_value <= 0 or np.isnan(depth_value):
            return None
        
        # pinhole camera model
        x = (pixel_x - self.camera_intrinsics['cx']) * depth_value / self.camera_intrinsics['fx']
        y = (pixel_y - self.camera_intrinsics['cy']) * depth_value / self.camera_intrinsics['fy']
        z = depth_value
        
        return (x, y, z)
    
    def detect_pinch(self, hand_landmarks):
        # fingertip indices: thumb, index, middle, ring, pinky
        fingertip_indices = [4, 8, 12, 16, 20]
        fingertips = [hand_landmarks[i] for i in fingertip_indices]
        
        total_dist = 0.0
        num_pairs = 0
        
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                tip_i = fingertips[i]
                tip_j = fingertips[j]
                
                dist = np.sqrt(
                    (tip_i.x - tip_j.x)**2 + 
                    (tip_i.y - tip_j.y)**2 +
                    (tip_i.z - tip_j.z)**2
                )
                total_dist += dist
                num_pairs += 1
        
        avg_fingertip_dist = total_dist / num_pairs if num_pairs > 0 else 0.0
        
        # hand size for scale invariance
        wrist = hand_landmarks[0]
        middle_mcp = hand_landmarks[9]
        
        hand_size = np.sqrt(
            (wrist.x - middle_mcp.x)**2 + 
            (wrist.y - middle_mcp.y)**2 +
            (wrist.z - middle_mcp.z)**2
        )
        
        if hand_size > 0:
            normalized_spread = avg_fingertip_dist / hand_size
            is_fist = normalized_spread < self.pinch_threshold
            return is_fist, normalized_spread
        
        return False, 0.0
    
    def calculate_palm_centroid(self, hand_landmarks, image_shape):
        palm_indices = [0, 5, 9, 13, 17]
        
        x_norm = np.mean([hand_landmarks[i].x for i in palm_indices])
        y_norm = np.mean([hand_landmarks[i].y for i in palm_indices])
        
        height, width = image_shape[:2]
        x_pixel = int(x_norm * width)
        y_pixel = int(y_norm * height)
        
        return x_pixel, y_pixel, x_norm, y_norm
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            result = self.detector.detect(rgb_image)
            annotated_image = draw_landmarks_on_image(rgb_image, result)
            
            is_pinching = False
            normalized_spread = 0.0
            palm_x_pixel, palm_y_pixel = 0, 0
            palm_x_norm, palm_y_norm = 0.0, 0.0
            num_hands = len(result.hand_landmarks)
            
            if num_hands > 0:
                is_pinching, normalized_spread = self.detect_pinch(result.hand_landmarks[0])
                palm_x_pixel, palm_y_pixel, palm_x_norm, palm_y_norm = self.calculate_palm_centroid(
                    result.hand_landmarks[0], 
                    rgb_image.shape
                )
                
                handedness_str = ', '.join([
                    f"{h[0].category_name} ({h[0].score:.2f})"
                    for h in result.handedness
                ])
                hand_status = "FIST" if is_pinching else "OPEN"
                self.get_logger().info(
                    f'hand: {handedness_str} | spread: {normalized_spread:.4f} | thresh: {self.pinch_threshold:.4f} | {hand_status}',
                    throttle_duration_sec=0.5
                )
            
            pinch_msg = Bool()
            pinch_msg.data = bool(is_pinching)
            self.pinch_pub.publish(pinch_msg)
            
            hand_text = "PINCHING" if is_pinching else "OPEN"
            hand_color = (0, 255, 0) if is_pinching else (255, 0, 0)
            
            cv2.putText(annotated_image, hand_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_color, 2, cv2.LINE_AA)
            
            if num_hands > 0:
                cv2.circle(annotated_image, (palm_x_pixel, palm_y_pixel), 10, (0, 0, 255), -1)
                cv2.circle(annotated_image, (palm_x_pixel, palm_y_pixel), 8, (255, 255, 255), 2)
            
            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
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
                
                self.annotated_camera_info_pub.publish(camera_info_msg)
            
        except Exception as e:
            self.get_logger().error(f'error processing image: {e}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = HandLandmarkerNode()
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

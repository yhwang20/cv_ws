#!/usr/bin/env python3
"""
ROS 2 Node for real-time hand landmark detection.

Subscribes to /image topic and publishes annotated images to /image_annotated.
"""

import os
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory

# Import custom messages (will be generated after build)
try:
    from hand_landmarker.msg import PinchStatus, HandPosition
except ImportError:
    # Fallback during first build
    PinchStatus = None
    HandPosition = None

from hand_landmarker import HandLandmarkDetector, draw_landmarks_on_image


class HandLandmarkerNode(Node):
    """ROS 2 node for hand landmark detection."""
    
    def __init__(self):
        """Initialize the hand landmarker node."""
        super().__init__('hand_landmarker')
        
        # Get default model path from package installation
        try:
            package_share_dir = get_package_share_directory('hand_landmarker')
            default_model_path = os.path.join(package_share_dir, 'models', 'hand_landmarker.task')
        except Exception as e:
            self.get_logger().warn(f'Could not get package share directory: {e}')
            default_model_path = 'hand_landmarker.task'
        
        # Declare parameters
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('min_detection_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        
        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        min_detection_confidence = self.get_parameter('min_detection_confidence').get_parameter_value().double_value
        min_tracking_confidence = self.get_parameter('min_tracking_confidence').get_parameter_value().double_value
        
        # Initialize cv_bridge
        self.bridge = CvBridge()
        
        # Initialize hand landmark detector
        try:
            # Import DetectorConfig
            from hand_landmarker import DetectorConfig
            
            # Create configuration
            config = DetectorConfig(
                model_path=model_path,
                min_tracking_confidence=min_tracking_confidence,
                min_hand_detection_confidence=min_detection_confidence
            )
            
            self.detector = HandLandmarkDetector(config)
            self.get_logger().info(f'Hand landmark detector initialized with model: {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize detector: {e}')
            raise
        
        # Create subscriber to /image topic
        self.image_sub = self.create_subscription(
            Image,
            '/image',
            self.image_callback,
            10
        )
        
        # Create publisher for annotated images
        self.image_pub = self.create_publisher(Image, '/image_annotated', 10)
        
        # Create publisher for pinch status (custom message)
        if PinchStatus is not None:
            self.pinch_pub = self.create_publisher(PinchStatus, '/is_pinching', 10)
        
        # Create publisher for palm position (custom message with pixel coordinates)
        if HandPosition is not None:
            self.hand_pos_pub = self.create_publisher(HandPosition, '/hand_pos', 10)
        
        # Pinch detection threshold (as ratio of hand size)
        self.declare_parameter('pinch_threshold', 0.40)  # 40% of hand size
        self.pinch_threshold = self.get_parameter('pinch_threshold').get_parameter_value().double_value
        
        self.get_logger().info('Hand landmarker node started')
        self.get_logger().info('Subscribing to: /image')
        self.get_logger().info('Publishing to: /image_annotated, /is_pinching, /hand_pos')
    
    def detect_pinch(self, hand_landmarks):
        """
        Detect if thumb tip and index finger tip are pinching.
        
        Thresholding explanation:
        1. Calculate 3D Euclidean distance between landmark 4 (thumb tip) and landmark 8 (index tip)
        2. Calculate hand size as distance between landmark 0 (wrist) and landmark 9 (middle finger MCP)
        3. Normalize: normalized_distance = thumb_index_distance / hand_size
        4. If normalized_distance < threshold, then PINCH detected
        
        This makes detection scale-invariant (works regardless of hand distance from camera).
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            tuple: (bool, float) - (is_pinching, normalized_distance)
        """
        # Get landmark positions
        thumb_tip = hand_landmarks[4]  # Thumb tip
        index_tip = hand_landmarks[8]  # Index finger tip
        wrist = hand_landmarks[0]      # Wrist
        middle_mcp = hand_landmarks[9] # Middle finger MCP (base)
        
        # Calculate distance between thumb and index tips (3D Euclidean distance)
        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 +
            (thumb_tip.z - index_tip.z)**2
        )
        
        # Calculate hand size (wrist to middle finger base) for scale invariance
        hand_size = np.sqrt(
            (wrist.x - middle_mcp.x)**2 + 
            (wrist.y - middle_mcp.y)**2 +
            (wrist.z - middle_mcp.z)**2
        )
        
        # Normalize distance by hand size
        if hand_size > 0:
            normalized_dist = thumb_index_dist / hand_size
            is_pinching = normalized_dist < self.pinch_threshold
            return is_pinching, normalized_dist
        
        return False, 0.0
    
    def calculate_palm_centroid(self, hand_landmarks, image_shape):
        """
        Calculate the centroid of the palm from key landmarks.
        
        Palm landmarks: 0 (wrist), 5 (index MCP), 9 (middle MCP), 13 (ring MCP), 17 (pinky MCP)
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            image_shape: (height, width) of the image for coordinate conversion
            
        Returns:
            tuple: (x_pixel, y_pixel, x_norm, y_norm) - pixel and normalized coordinates
        """
        # Palm landmarks indices
        palm_indices = [0, 5, 9, 13, 17]
        
        # Calculate centroid in normalized coordinates
        x_norm = np.mean([hand_landmarks[i].x for i in palm_indices])
        y_norm = np.mean([hand_landmarks[i].y for i in palm_indices])
        
        # Convert to pixel coordinates
        height, width = image_shape[:2]
        x_pixel = int(x_norm * width)
        y_pixel = int(y_norm * height)
        
        return x_pixel, y_pixel, x_norm, y_norm
    
    def image_callback(self, msg):
        """Process incoming image and publish annotated result."""
        try:
            # Convert ROS Image message to OpenCV image (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Detect hand landmarks
            result = self.detector.detect(rgb_image)
            
            # Draw landmarks on image
            annotated_image = draw_landmarks_on_image(rgb_image, result)
            
            # Check for pinch gesture and calculate palm position
            is_pinching = False
            normalized_dist = 0.0
            palm_x_pixel, palm_y_pixel = 0, 0
            palm_x_norm, palm_y_norm = 0.0, 0.0
            num_hands = len(result.hand_landmarks)
            
            if num_hands > 0:
                # Check pinch for the first detected hand
                is_pinching, normalized_dist = self.detect_pinch(result.hand_landmarks[0])
                
                # Calculate palm centroid
                palm_x_pixel, palm_y_pixel, palm_x_norm, palm_y_norm = self.calculate_palm_centroid(
                    result.hand_landmarks[0], 
                    rgb_image.shape
                )
                
                # Log detection results with debug info
                handedness_str = ', '.join([
                    f"{h[0].category_name} ({h[0].score:.2f})"
                    for h in result.handedness
                ])
                pinch_status = "PINCHING" if is_pinching else "NOT PINCHING"
                self.get_logger().info(
                    f'Hand: {handedness_str} | Landmarks 4-8 normalized dist: {normalized_dist:.4f} | Threshold: {self.pinch_threshold:.4f} | Status: {pinch_status} | Palm: ({palm_x_pixel}, {palm_y_pixel})',
                    throttle_duration_sec=0.5
                )
            
            # Publish pinch status (custom message with 'pinch' field)
            if PinchStatus is not None:
                pinch_msg = PinchStatus()
                pinch_msg.pinch = bool(is_pinching)
                self.pinch_pub.publish(pinch_msg)
            
            # Publish hand position in pixel coordinates (custom message)
            if HandPosition is not None:
                hand_pos_msg = HandPosition()
                hand_pos_msg.x = int(palm_x_pixel)
                hand_pos_msg.y = int(palm_y_pixel)
                hand_pos_msg.z = 0
                self.hand_pos_pub.publish(hand_pos_msg)
            
            # Add pinch status text to image
            pinch_text = "PINCH" if is_pinching else "NO PINCH"
            pinch_color = (0, 255, 0) if is_pinching else (255, 0, 0)  # Green for pinch, red for no pinch
            
            cv2.putText(
                annotated_image,
                pinch_text,
                (10, 20),  # Top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                pinch_color,
                2,  # Thickness
                cv2.LINE_AA
            )
            
            # Add palm position text (blue)
            palm_text = f"Hand Position: [{palm_x_pixel}, {palm_y_pixel}, 0]"
            cv2.putText(
                annotated_image,
                palm_text,
                (10, 45),  # Below pinch text
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # Font scale
                (0, 0, 255),  # Blue in RGB
                2,  # Thickness
                cv2.LINE_AA
            )
            
            # Draw blue dot at palm centroid
            if num_hands > 0:
                cv2.circle(
                    annotated_image,
                    (palm_x_pixel, palm_y_pixel),
                    10,  # Radius
                    (0, 0, 255),  # Blue in RGB
                    -1  # Filled circle
                )
                # Add white border for visibility
                cv2.circle(
                    annotated_image,
                    (palm_x_pixel, palm_y_pixel),
                    8,  # Radius
                    (255, 255, 255),  # White border in RGB
                    2  # Border thickness
                )
            
            # Convert back to BGR for ROS
            annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            # Convert to ROS Image message
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding='bgr8')
            annotated_msg.header = msg.header  # Keep the same header (timestamp, frame_id)
            
            # Publish annotated image
            self.image_pub.publish(annotated_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        node = HandLandmarkerNode()
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


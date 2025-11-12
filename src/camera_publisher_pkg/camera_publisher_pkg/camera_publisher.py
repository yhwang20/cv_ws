#!/usr/bin/env python3
"""
ROS 2 Node for camera publisher.
"""

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPublisher(Node):
    """Camera publisher node."""
    
    def __init__(self):
        """Initialize the camera publisher node."""
        super().__init__('camera_publisher')
        
        # Declare parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('fps', 30)
        
        # Get parameters
        self.camera_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        fps = self.get_parameter('fps').get_parameter_value().integer_value
        
        # Create publisher
        self.image_pub = self.create_publisher(Image, '/image', 10)
        
        # Initialize cv_bridge
        self.bridge = CvBridge()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            self.get_logger().error(f'Failed to open camera {self.camera_id}')
            raise RuntimeError(f'Cannot open camera {self.camera_id}')
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Create timer for publishing at specified FPS
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.publish_image)
        
        self.get_logger().info(f'Camera publisher started on /image topic')
        self.get_logger().info(f'Camera ID: {self.camera_id}, FPS: {fps}')
    
    def publish_image(self):
        """Capture and publish a single image frame."""
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warn('Failed to read frame from camera')
            return
        
        try:
            # Convert OpenCV image to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'camera_frame'
            
            # Publish the image
            self.image_pub.publish(ros_image)
            
        except Exception as e:
            self.get_logger().error(f'Error converting/publishing image: {e}')
    
    def destroy_node(self):
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()
        self.get_logger().info('Camera publisher shutdown')
        super().destroy_node()


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    try:
        publisher = CameraPublisher()
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        if 'publisher' in locals():
            publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPublisher(Node):
    
    def __init__(self):
        super().__init__('camera_publisher')
        
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('fps', 30)
        
        self.camera_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        fps = self.get_parameter('fps').get_parameter_value().integer_value
        
        self.image_pub = self.create_publisher(Image, '/image', 10)
        self.bridge = CvBridge()
        
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            self.get_logger().error(f'failed to open camera {self.camera_id}')
            raise RuntimeError(f'cannot open camera {self.camera_id}')
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.publish_image)
        
        self.get_logger().info(f'camera publisher started on /image')
    
    def publish_image(self):
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warn('failed to read frame')
            return
        
        try:
            ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'camera_frame'
            
            self.image_pub.publish(ros_image)
            
        except Exception as e:
            self.get_logger().error(f'error publishing image: {e}')
    
    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
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

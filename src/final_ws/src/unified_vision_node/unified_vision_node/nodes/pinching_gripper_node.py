#!/usr/bin/env python3
"""Pinching Gripper Node - Python port of the C++ pinching_gripper node."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64MultiArray


class PinchingGripperNode(Node):
    """
    Node that subscribes to /is_pinching and controls gripper.
    
    Functionality:
    - Subscribes to /is_pinching (Bool)
    - Publishes gripper commands to /gripper_controller/commands
    - When pinching is True: closes gripper (0.8)
    - When pinching is False: opens gripper (0.0)
    """
    
    def __init__(self):
        super().__init__('pinching_gripper_node')
        
        # Create publisher for gripper commands
        self.gripper_pub = self.create_publisher(
            Float64MultiArray,
            '/gripper_controller/commands',
            10
        )
        
        # Subscribe to is_pinching topic
        self.subscription = self.create_subscription(
            Bool,
            '/is_pinching',
            self.pinching_callback,
            10
        )
        
        self.get_logger().info('Pinching Gripper Node initialized')
        self.get_logger().info('Listening to /is_pinching topic...')
        self.get_logger().info('Gripper open = 0.0, closed = 0.8')
    
    def pinching_callback(self, msg):
        """Handle incoming pinching status messages."""
        gripper_cmd = Float64MultiArray()
        
        if msg.data:
            # Pinching is true - close gripper
            gripper_cmd.data = [0.8]
            self.get_logger().info('Pinching detected - closing gripper (0.8)')
        else:
            # Pinching is false - open gripper
            gripper_cmd.data = [0.0]
            self.get_logger().info('No pinching - opening gripper (0.0)')
        
        # Publish gripper command
        self.gripper_pub.publish(gripper_cmd)


def main(args=None):
    """Main entry point for the pinching gripper node."""
    rclpy.init(args=args)
    node = PinchingGripperNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


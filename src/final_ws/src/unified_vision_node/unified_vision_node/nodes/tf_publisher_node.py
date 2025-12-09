#!/usr/bin/env python3
"""TF Publisher Node - Python port of the C++ publish_transform node."""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster, Buffer, TransformListener
from tf2_ros import TransformException
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation as R
import numpy as np


class TFPublisherNode(Node):
    """
    Node that subscribes to /wrist_in_shoulder and publishes target transform.
    
    Functionality:
    - Subscribes to /wrist_in_shoulder (PoseStamped)
    - On first message, looks up ooks up transform from world to tool0 (tool_start) up transform from world to tool0 (tool_start)ransform from world to tool0 (tool_start)
    - Calculates delta transform between consecutive wrist poses
    - Publishes static transform: target = tool_start * deltaT
    """
    
    def __init__(self):
        super().__init__('tf_publisher_node')
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        # State variables
        self.tool_start_initialized = False
        self.first_wrist_pose_received = False
        self.tool_start_translation = None
        self.tool_start_rotation = None
        self.prior_wrist_pose = None
        
        # Current target transform (accumulates over time)
        self.current_target_translation = None
        self.current_target_rotation = None
        
        # Subscribe to wrist_in_shoulder topic
        self.subscription = self.create_subscription(
            PoseStamped,
            '/wrist_in_shoulder',
            self.wrist_callback,
            10
        )
        
        self.get_logger().info('TF Publisher Node initialized')
        self.get_logger().info('Waiting for /wrist_in_shoulder messages...')
    
    def wrist_callback(self, msg):
        """Handle incoming wrist pose messages."""
        # On first message, initialize tool_start transform
        if not self.tool_start_initialized:
            self.initialize_tool_start()
        
        # Store current wrist pose and return on first reception
        if not self.first_wrist_pose_received:
            self.prior_wrist_pose = msg
            self.first_wrist_pose_received = True
            self.get_logger().info('Received first wrist pose')
            return
        
        # Use inverse of wrist_in_shoulder poses
        current_wrist_tf = self.pose_to_transform_inverse(msg.pose)
        prior_wrist_tf = self.pose_to_transform_inverse(self.prior_wrist_pose.pose)
        
        # Calculate deltaT (relative transform between current and prior wrist poses)
        # deltaT = prior_wrist^(-1) * current_wrist
        delta_t = self.compose_transforms(
            self.invert_transform(prior_wrist_tf),
            current_wrist_tf
        )
        
        # Remap from hand frame to target frame
        remapped_delta_t = self.remap_hand_to_target_frame(delta_t)
        
        # Scale deltaT down
        scaled_delta_t = self.scale_transform(remapped_delta_t, 0.15)
        
        # Update prior wrist pose for next iteration
        self.prior_wrist_pose = msg
        
        # If tool_start is initialized, accumulate delta and publish target
        if self.tool_start_initialized:
            # Accumulate: target = current_target * scaled_deltaT
            # This builds up all motion over time instead of resetting to tool_start
            current_target_tf = (self.current_target_translation, self.current_target_rotation)
            new_target = self.compose_transforms(current_target_tf, scaled_delta_t)
            
            # Update current target for next iteration
            self.current_target_translation, self.current_target_rotation = new_target
            
            # Publish static transform from world to target
            self.publish_static_transform(new_target)
    
    def initialize_tool_start(self):
        """Look up and store the transform from world to tool0."""
        try:
            # Look up transform from world to tool0
            transform_stamped = self.tf_buffer.lookup_transform(
                'world',
                'tool0',
                rclpy.time.Time()
            )
            
            # Store as tool_start
            self.tool_start_translation = np.array([
                transform_stamped.transform.translation.x,
                transform_stamped.transform.translation.y,
                transform_stamped.transform.translation.z
            ])
            self.tool_start_rotation = np.array([
                transform_stamped.transform.rotation.x,
                transform_stamped.transform.rotation.y,
                transform_stamped.transform.rotation.z,
                transform_stamped.transform.rotation.w
            ])
            
            # Initialize current target to match tool_start
            self.current_target_translation = self.tool_start_translation.copy()
            self.current_target_rotation = self.tool_start_rotation.copy()
            
            self.tool_start_initialized = True
            
            self.get_logger().info('Tool start transform initialized from world to tool0')
            self.get_logger().info(
                f'Position: [{self.tool_start_translation[0]:.3f}, '
                f'{self.tool_start_translation[1]:.3f}, '
                f'{self.tool_start_translation[2]:.3f}]'
            )
            self.get_logger().info(
                f'Orientation: [{self.tool_start_rotation[0]:.3f}, '
                f'{self.tool_start_rotation[1]:.3f}, '
                f'{self.tool_start_rotation[2]:.3f}, '
                f'{self.tool_start_rotation[3]:.3f}]'
            )
            
        except TransformException as ex:
            self.get_logger().warn(
                f'Could not get transform from world to tool0: {ex}'
            )
            self.get_logger().warn('Will retry on next callback...')
    
    def pose_to_transform_inverse(self, pose):
        """
        Convert a Pose message to an inverse transform (translation, quaternion).
        
        Returns:
            tuple: (translation_array, quaternion_array) representing the inverse transform
        """
        # Create forward transform
        translation = np.array([pose.position.x, pose.position.y, pose.position.z])
        quaternion = np.array([
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ])
        
        # Compute inverse
        return self.invert_transform((translation, quaternion))
    
    def invert_transform(self, transform):
        """
        Invert a transform represented as (translation, quaternion).
        
        Args:
            transform: tuple of (translation_array, quaternion_array)
        
        Returns:
            tuple: inverted (translation_array, quaternion_array)
        """
        translation, quaternion = transform
        
        # Invert rotation
        rot = R.from_quat(quaternion)
        rot_inv = rot.inv()
        quat_inv = rot_inv.as_quat()
        
        # Invert translation: -R^T * t
        trans_inv = -rot_inv.apply(translation)
        
        return (trans_inv, quat_inv)
    
    def compose_transforms(self, tf1, tf2):
        """
        Compose two transforms: result = tf1 * tf2.
        
        Args:
            tf1: tuple of (translation_array, quaternion_array)
            tf2: tuple of (translation_array, quaternion_array)
        
        Returns:
            tuple: composed (translation_array, quaternion_array)
        """
        t1, q1 = tf1
        t2, q2 = tf2
        
        # Compose rotations
        rot1 = R.from_quat(q1)
        rot2 = R.from_quat(q2)
        rot_result = rot1 * rot2
        quat_result = rot_result.as_quat()
        
        # Compose translations: t_result = t1 + R1 * t2
        trans_result = t1 + rot1.apply(t2)
        
        return (trans_result, quat_result)
    
    def remap_hand_to_target_frame(self, transform):
        """
        Remap hand frame to target frame with axis transformation:
        hand +x → target -z
        hand +y → target +y
        hand +z → target -x
        
        Args:
            transform: tuple of (translation_array, quaternion_array) in hand frame
        
        Returns:
            tuple: remapped (translation_array, quaternion_array) in target frame
        """
        translation, quaternion = transform
        
        # Remap translation components
        # target = [-hand_z, hand_y, -hand_x]
        remapped_translation = np.array([
            -translation[2],  # target_x = -hand_z
            translation[1],   # target_y = hand_y
            -translation[0]   # target_z = -hand_x
        ])
        
        # Define rotation matrix for coordinate transformation
        # This matrix transforms vectors from hand frame to target frame
        rotation_matrix = np.array([
            [0,  0, -1],  # target_x row
            [0,  1,  0],  # target_y row
            [-1, 0,  0]   # target_z row
        ])
        
        # Convert hand frame rotation to matrix, apply coordinate transform, convert back
        hand_rot = R.from_quat(quaternion)
        hand_rot_matrix = hand_rot.as_matrix()
        
        # Transform rotation: R_target = R_coord * R_hand * R_coord^T
        target_rot_matrix = rotation_matrix @ hand_rot_matrix @ rotation_matrix.T
        target_rot = R.from_matrix(target_rot_matrix)
        remapped_quaternion = target_rot.as_quat()
        
        return (remapped_translation, remapped_quaternion)
    
    def scale_transform(self, transform, scale_factor):
        """
        Scale a transform by a given factor.
        
        Args:
            transform: tuple of (translation_array, quaternion_array)
            scale_factor: scaling factor (e.g., 0.1 for 1/10)
        
        Returns:
            tuple: scaled (translation_array, quaternion_array)
        """
        translation, quaternion = transform
        
        # Scale translation
        scaled_translation = translation * scale_factor
        
        # Scale rotation: convert to axis-angle, scale angle, convert back
        rot = R.from_quat(quaternion)
        rotvec = rot.as_rotvec()  # axis-angle representation
        scaled_rotvec = rotvec * scale_factor
        scaled_rot = R.from_rotvec(scaled_rotvec)
        scaled_quaternion = scaled_rot.as_quat()
        
        return (scaled_translation, scaled_quaternion)
    
    def publish_static_transform(self, target):
        """
        Publish the target transform from world to target frame.
        
        Args:
            target: tuple of (translation_array, quaternion_array)
        """
        translation, quaternion = target
        
        transform_stamped = TransformStamped()
        
        # Set header
        transform_stamped.header.stamp = self.get_clock().now().to_msg()
        transform_stamped.header.frame_id = 'world'
        transform_stamped.child_frame_id = 'target'
        
        # Set translation
        transform_stamped.transform.translation.x = float(translation[0])
        transform_stamped.transform.translation.y = float(translation[1])
        transform_stamped.transform.translation.z = float(translation[2])
        
        # Set rotation
        transform_stamped.transform.rotation.x = float(quaternion[0])
        transform_stamped.transform.rotation.y = float(quaternion[1])
        transform_stamped.transform.rotation.z = float(quaternion[2])
        transform_stamped.transform.rotation.w = float(quaternion[3])
        
        # Publish static transform
        self.static_tf_broadcaster.sendTransform(transform_stamped)


def main(args=None):
    """Main entry point for the TF publisher node."""
    rclpy.init(args=args)
    node = TFPublisherNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


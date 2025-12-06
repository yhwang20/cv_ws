"""Launch file for unified vision node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for unified vision system."""
    
    # Declare launch arguments for pinch model
    pinch_threshold_arg = DeclareLaunchArgument(
        'pinch_threshold',
        default_value='0.65',
        description='Threshold for pinch detection (lower = more sensitive)'
    )
    
    min_hand_detection_conf_arg = DeclareLaunchArgument(
        'min_hand_detection_confidence',
        default_value='0.5',
        description='Minimum confidence for hand detection'
    )
    
    min_hand_tracking_conf_arg = DeclareLaunchArgument(
        'min_hand_tracking_confidence',
        default_value='0.5',
        description='Minimum confidence for hand tracking'
    )
    
    # Declare launch arguments for location model
    min_pose_detection_conf_arg = DeclareLaunchArgument(
        'min_pose_detection_confidence',
        default_value='0.5',
        description='Minimum confidence for pose detection'
    )
    
    min_pose_tracking_conf_arg = DeclareLaunchArgument(
        'min_pose_tracking_confidence',
        default_value='0.5',
        description='Minimum confidence for pose tracking'
    )
    
    pose_model_complexity_arg = DeclareLaunchArgument(
        'pose_model_complexity',
        default_value='2',
        description='Pose model complexity (0, 1, or 2)'
    )
    
    hand_side_arg = DeclareLaunchArgument(
        'hand_side',
        default_value='RIGHT',
        description='Which hand to track (RIGHT or LEFT)'
    )
    
    depth_window_size_arg = DeclareLaunchArgument(
        'depth_window_size',
        default_value='20',
        description='Rolling average window size for depth smoothing'
    )
    
    # Create unified vision node
    unified_vision_node = Node(
        package='unified_vision_node',
        executable='unified_vision_node',
        name='unified_vision',
        output='screen',
        parameters=[{
            'pinch_threshold': LaunchConfiguration('pinch_threshold'),
            'min_hand_detection_confidence': LaunchConfiguration('min_hand_detection_confidence'),
            'min_hand_tracking_confidence': LaunchConfiguration('min_hand_tracking_confidence'),
            'min_pose_detection_confidence': LaunchConfiguration('min_pose_detection_confidence'),
            'min_pose_tracking_confidence': LaunchConfiguration('min_pose_tracking_confidence'),
            'pose_model_complexity': LaunchConfiguration('pose_model_complexity'),
            'hand_side': LaunchConfiguration('hand_side'),
            'depth_window_size': LaunchConfiguration('depth_window_size'),
        }],
        emulate_tty=True,
    )
    
    # Static transform publisher: map -> camera_link
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_map_to_camera',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'camera_link']
    )
    
    return LaunchDescription([
        pinch_threshold_arg,
        min_hand_detection_conf_arg,
        min_hand_tracking_conf_arg,
        min_pose_detection_conf_arg,
        min_pose_tracking_conf_arg,
        pose_model_complexity_arg,
        hand_side_arg,
        depth_window_size_arg,
        unified_vision_node,
        static_tf_node,
    ])


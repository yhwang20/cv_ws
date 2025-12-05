"""Launch file for pose detector node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for pose detector."""
    
    # Declare launch arguments
    min_detection_confidence_arg = DeclareLaunchArgument(
        'min_detection_confidence',
        default_value='0.5',
        description='Minimum confidence for pose detection (0.0-1.0)'
    )
    
    min_tracking_confidence_arg = DeclareLaunchArgument(
        'min_tracking_confidence',
        default_value='0.5',
        description='Minimum confidence for pose tracking (0.0-1.0)'
    )
    
    model_complexity_arg = DeclareLaunchArgument(
        'model_complexity',
        default_value='2',
        description='Pose model complexity (0, 1, or 2 - higher is more accurate but slower)'
    )
    
    hand_side_arg = DeclareLaunchArgument(
        'hand_side',
        default_value='RIGHT',
        description='Which hand to track (RIGHT or LEFT)'
    )
    
    depth_window_size_arg = DeclareLaunchArgument(
        'depth_window_size',
        default_value='5',
        description='Rolling average window size for depth smoothing'
    )
    
    # Create pose detector node
    pose_detector_node = Node(
        package='hand_body_detection',
        executable='pose_detector_node',
        name='pose_detector',
        output='screen',
        parameters=[{
            'min_detection_confidence': LaunchConfiguration('min_detection_confidence'),
            'min_tracking_confidence': LaunchConfiguration('min_tracking_confidence'),
            'model_complexity': LaunchConfiguration('model_complexity'),
            'hand_side': LaunchConfiguration('hand_side'),
            'depth_window_size': LaunchConfiguration('depth_window_size'),
        }],
        emulate_tty=True,
    )
    
    return LaunchDescription([
        min_detection_confidence_arg,
        min_tracking_confidence_arg,
        model_complexity_arg,
        hand_side_arg,
        depth_window_size_arg,
        pose_detector_node,
    ])


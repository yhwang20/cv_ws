"""Launch file for hand landmarker with camera publisher."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description."""
    
    # Declare launch arguments
    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Camera device ID'
    )
    
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30',
        description='Target frames per second'
    )
    
    # Camera publisher node
    camera_node = Node(
        package='camera_publisher_pkg',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen',
        parameters=[{
            'camera_id': LaunchConfiguration('camera_id'),
            'fps': LaunchConfiguration('fps'),
        }]
    )
    
    # Hand landmarker node
    hand_landmarker_node = Node(
        package='hand_landmarker',
        executable='hand_landmarker_node',
        name='hand_landmarker',
        output='screen',
        parameters=[{
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
        }]
    )
    
    return LaunchDescription([
        camera_id_arg,
        fps_arg,
        camera_node,
        hand_landmarker_node,
    ])


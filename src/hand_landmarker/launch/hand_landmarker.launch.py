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
    
    # Hand landmarker node (for MediaPipe pinch detection)
    hand_landmarker_node = Node(
        package='hand_landmarker',
        executable='hand_landmarker_node',
        name='hand_landmarker',
        output='screen',
        parameters=[{
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'debug_depth': True,  # Enable to see depth debug messages
        }]
    )

    # Hand and Body Detection node (for MediaPipe pose detection of wrist and shoulder)
    pose_detector_node = Node(
        package='hand_body_detection',
        executable='pose_detector_node',
        name='pose_detector',
        output='screen',
        parameters=[{
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'model_complexity': 2,
            'hand_side': 'RIGHT',
            'depth_window_size': 5,  # Rolling average window for depth smoothing
        }]
    )
    
    # Static transform publisher: map -> camera (identity transform)
    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_publisher_map_to_camera',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'camera_link']
    )
    
    return LaunchDescription([
        camera_id_arg,
        fps_arg,
        #camera_node,
        hand_landmarker_node,
        pose_detector_node,
        static_tf_node,
    ])


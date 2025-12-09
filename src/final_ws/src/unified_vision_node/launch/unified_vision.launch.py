"""Launch file for unified vision node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for unified vision system."""
    
    # Get package directory
    pkg_dir = get_package_share_directory('unified_vision_node')
    
    # Declare launch arguments for model paths
    hand_model_path_arg = DeclareLaunchArgument(
        'hand_landmarker_model_path',
        default_value='hand_landmarker.task',
        description='Path to hand landmarker model (relative to weights/ or absolute)'
    )
    
    gesture_model_path_arg = DeclareLaunchArgument(
        'gesture_recognizer_model_path',
        default_value='gesture_recognizer.task',
        description='Path to gesture recognizer model (relative to weights/ or absolute)'
    )
    
    custom_pinch_model_path_arg = DeclareLaunchArgument(
        'custom_pinch_model_path',
        default_value='hand_gesture_model.pth',
        description='Path to custom pinch model (relative to weights/ or absolute)'
    )
    
    yolo_pose_model_path_arg = DeclareLaunchArgument(
        'yolo_pose_model_path',
        default_value='yolo11n-pose.pt',
        description='Path to YOLO pose model (relative to weights/ or absolute)'
    )
    
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
    
    # PINCH MODEL SELECTION
    pinch_model_arg = DeclareLaunchArgument(
        'pinch',
        default_value='mediapipegesture',
        description='Pinch detection model: custom, mediapipefinger, or mediapipegesture'
    )
    
    # POSE MODEL SELECTION
    pose_model_arg = DeclareLaunchArgument(
        'pose',
        default_value='mediapipe',
        description='Pose detection model: mediapipe or yolo'
    )
    
    # Create unified vision node
    unified_vision_node = Node(
        package='unified_vision_node',
        executable='unified_vision_node',
        name='unified_vision',
        output='screen',
        parameters=[{
            'hand_landmarker_model_path': LaunchConfiguration('hand_landmarker_model_path'),
            'gesture_recognizer_model_path': LaunchConfiguration('gesture_recognizer_model_path'),
            'custom_pinch_model_path': LaunchConfiguration('custom_pinch_model_path'),
            'yolo_pose_model_path': LaunchConfiguration('yolo_pose_model_path'),
            'pinch_threshold': LaunchConfiguration('pinch_threshold'),
            'min_hand_detection_confidence': LaunchConfiguration('min_hand_detection_confidence'),
            'min_hand_tracking_confidence': LaunchConfiguration('min_hand_tracking_confidence'),
            'min_pose_detection_confidence': LaunchConfiguration('min_pose_detection_confidence'),
            'min_pose_tracking_confidence': LaunchConfiguration('min_pose_tracking_confidence'),
            'pose_model_complexity': LaunchConfiguration('pose_model_complexity'),
            'hand_side': LaunchConfiguration('hand_side'),
            'depth_window_size': LaunchConfiguration('depth_window_size'),
            'pinch': LaunchConfiguration('pinch'),
            'pose': LaunchConfiguration('pose'),
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
    
    # RealSense camera node
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        output='screen',
        parameters=[{
            'enable_color': True,
            'enable_depth': True,
            'align_depth.enable': True,
            'enable_infra1': False,
            'enable_infra2': False,
        }],
        emulate_tty=True,
    )
    
    # TF publisher node
    tf_publisher_node = Node(
        package='unified_vision_node',
        executable='tf_publisher_node',
        name='tf_publisher',
        output='screen',
    )

    # Pinching gripper node
    pinching_gripper_node = Node(
        package='unified_vision_node',
        executable='pinching_gripper_node',
        name='pinching_gripper',
        output='screen',
    )

    # Return launch description
    return LaunchDescription([
        hand_model_path_arg,
        gesture_model_path_arg,
        custom_pinch_model_path_arg,
        yolo_pose_model_path_arg,
        pinch_threshold_arg,
        min_hand_detection_conf_arg,
        min_hand_tracking_conf_arg,
        min_pose_detection_conf_arg,
        min_pose_tracking_conf_arg,
        pose_model_complexity_arg,
        hand_side_arg,
        depth_window_size_arg,
        pinch_model_arg,
        pose_model_arg,
        # realsense_node,
        unified_vision_node,
        tf_publisher_node,
        pinching_gripper_node,
        static_tf_node,
        # rviz_node,
    ])


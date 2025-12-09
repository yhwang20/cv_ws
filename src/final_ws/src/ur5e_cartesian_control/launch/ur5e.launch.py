from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import EnvironmentVariable
import os

# FAKE HARDWARE:
    # ROS_DOMAIN_ID=7  ros2 launch ur5e_cartesian_control ur5e.launch.py use_fake:=true headless_mode:=false

# REAL HARDWARE:
    # ROS_DOMAIN_ID=7  ros2 launch ur5e_cartesian_control ur5e.launch.py use_fake:=false headless_mode:=true

# CONTROLLERS I WANT TO BE ACTIVE:
    #joint_state_broadcaster
    #cartesian_controller
    #target_pose_broadcaster
    #robot_pose_broadcaster


def generate_launch_description():

    # command line arguments
    use_fake_hardware = DeclareLaunchArgument('use_fake', default_value='true')
    target_x = DeclareLaunchArgument('x', default_value='-0.5')
    target_y = DeclareLaunchArgument('y', default_value='0.0')
    target_z = DeclareLaunchArgument('z', default_value='0.4')
    robot_ip_arg = DeclareLaunchArgument('robot_ip', default_value='172.22.22.2')
    headless_mode = DeclareLaunchArgument('headless_mode', default_value='false')

    use_fake_hardware_value = LaunchConfiguration('use_fake')
    target_x_value = LaunchConfiguration('x')
    target_y_value = LaunchConfiguration('y')
    target_z_value = LaunchConfiguration('z')
    robot_ip = LaunchConfiguration('robot_ip')
    headless_mode = LaunchConfiguration('headless_mode')

    script_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "ros_control.urscript"]
    )

    input_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_input_recipe.txt"]
    )

    output_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_output_recipe.txt"]
    )

    robot_urdf = PathJoinSubstitution([FindPackageShare("ur5e_cartesian_control"), "urdf", "ur5e_cartesian_with_gripper.urdf.xacro"])
    initial_positions = PathJoinSubstitution([FindPackageShare("ur5e_cartesian_control"), "config", "initial_positions.yaml"])
    
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            robot_urdf,    
            " ",
            "use_fake_hardware:=", use_fake_hardware_value,
            " ",
            "fake_sensor_commands:=false",
            " ",
            "robot_ip:=", robot_ip,
            " ",
            "tf_prefix:=""",
            " ",
            "headless_mode:=", headless_mode, " ",
            "script_filename:=", script_filename, " ",
            "input_recipe_filename:=", input_recipe_filename, " ",
            "output_recipe_filename:=", output_recipe_filename, " ",
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    sensor_config = PathJoinSubstitution([FindPackageShare("se3_sensor_driver"), "config", "se3_sensor.yaml"])
    control_config = PathJoinSubstitution([FindPackageShare("ur5e_cartesian_control"), "config", "cartesian_control.yaml"])



    robot_joint_publisher_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        output="both",
        parameters=[
            robot_description,
            {"use_gui": True},
            {
                "zeros": {
                    "shoulder_pan_joint": 0,
                    "shoulder_lift_joint": -1.5708,
                    "elbow_joint": -1.5708,
                    "wrist_1_joint": -1.5708,
                    "wrist_2_joint": 1.5708,
                    "wrist_3_joint": 0.0
                }
            }
        ],
        condition=IfCondition(use_fake_hardware_value)  # don't run on real robot
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description],
        output="screen"
    )

    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robot_description,
            control_config,
            sensor_config
        ],
        output="screen",
    )

    joint_state_broadcaster_node = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    cartesian_controller_node = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["cartesian_controller", "--controller-manager", "/controller_manager"],
        output="screen"
    )

    gripper_controller_node = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
        output="screen",
        condition=IfCondition(use_fake_hardware_value)  # only in fake mode
    )

    # # sensor frame wrt world
    # sensor_static_transform = Node(
    #     package="tf2_ros",
    #     executable="static_transform_publisher",
    #     arguments=["-0.25", "0.25", "0", "0", "0", "0", "world", "sensor"],
    #     output="screen"
    # )

    # target frame wrt world (target "sensor" spawns wrt this)
    target_static_transform = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[target_x_value, target_y_value, target_z_value, "0.707", "0.707", "0.000", "0.000", "world", "target"],
        output="screen"
    )
    
    # broadcast robot "sensor"
    robot_pose_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["robot_pose_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen"
    )

    # broadcast target "sensor"
    target_pose_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["target_pose_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen"
    )

   # world = map 
    world_static_transform = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "map", "world"],
        output="screen"
    )

    rviz_config = PathJoinSubstitution([FindPackageShare('ur5e_cartesian_control'), 'config', 'ur5e_with_sensor.rviz'])

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
    )

    tf_socket_server_node = Node(
        package="se3_sensor_driver",
        executable="tf_socket_server",
        name="tf_socket_server",
        output="screen",
        parameters=[
            {"port": 12345},
            {"source_frame": "world"},
            {"robot_frame": "tool0"},
            {"target_frame": "target"}
        ]
    )





    return LaunchDescription([
        use_fake_hardware,
        target_x,
        target_y,
        target_z,
        robot_ip_arg,

        # robot_joint_publisher_node,  # DISABLED - conflicts with ros2_control joint_state_broadcaster
        robot_state_publisher_node,
        controller_manager_node,
        joint_state_broadcaster_node,
        # sensor_static_transform,
        target_static_transform,
        robot_pose_broadcaster_spawner,     # robot sensor
        target_pose_broadcaster_spawner,    # target sensor
        world_static_transform,
        rviz_node,
        tf_socket_server_node,
        cartesian_controller_node,
        gripper_controller_node

    ])

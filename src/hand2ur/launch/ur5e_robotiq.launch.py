import os

from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.substitutions import FindPackageShare
from ur_moveit_config.launch_common import load_yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution

from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description():
    
    # Declarations
    declared_arguments = [
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur5e", 
            description="Type/Series of used Robot",
        ),
        DeclareLaunchArgument(
            "robot_ip",
            default_value="172.22.22.2",
            description="IP address of the UR robot",
        ), 
        DeclareLaunchArgument(
            "safety_limits", 
            default_value="true",
            description="Enable the safety limits controller if true",
        ), 
        DeclareLaunchArgument(
            "safety_pos_margin",
            default_value="0.15",
            description="The margin to lower and upper limits in the safety controller.",
        ),
        DeclareLaunchArgument(
            "safety_k_position",
            default_value="20",
            description="k-position factor in the safety controller.",
        ),
        DeclareLaunchArgument(
            "runtime_config_package",
            default_value="asbr_description",
            description="Package with the controller's configuration in 'config' folder.",
        ),
         DeclareLaunchArgument(
            "description_package",
            default_value="asbr_description",
            description="Description package with custom/default robot URDF/XACRO files.",
        ),
        DeclareLaunchArgument(
            "description_file",
            default_value="asbr.urdf.xacro",
            description="The URDF/XACRO description file with the robot.",
        ),
        DeclareLaunchArgument(
            "controllers_file",
            default_value="controllers.yaml",
            description="YAML file with the conroller's configuration.",
        ),
        DeclareLaunchArgument(
            "moveit_config_package",
            default_value="asbr_moveit",
            description="Package of the MoveIt! configuration files.",
        ),
        DeclareLaunchArgument(
            "moveit_config_file",
            default_value="asbr.srdf.xacro",
            description="Package of the MoveIt! configuration files.",
        ),
        DeclareLaunchArgument(
            "kinematics_params_file",
            default_value="kinematics.yaml",
            description="The calibration configuration of the actual robot used.", 
        ),
        DeclareLaunchArgument(
            "joint_limits_file",
            default_value="joint_limits.yaml",
            description="MoveIt joint limits that augment or override the values from the URDF robot_description.",
        ),
        DeclareLaunchArgument(
            "tf_prefix",
            default_value="",
            description="Prefix of the joint names, useful for multi-robot setup",
        ),
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="true", 
            description="Start the robot with fake hardware.",
        ),
        DeclareLaunchArgument(
            "fake_sensor_commands", 
            default_value="false",
            description="Enable fake command interfaces for sensors used for simple simulations.",
        ),
        DeclareLaunchArgument(
            "controller_spawner_timeout", 
            default_value="10",
            description="Timeout used when spawning controllers.",
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="true",
            description="Launch RViz?",
        ),
        DeclareLaunchArgument(
            "headless_mode",
            default_value="true",
            description="Enable headless mode for real robot control.",
        ),
        DeclareLaunchArgument(
            "launch_dashboard_client",
            default_value="false",
            description="Launch Dashboard Client?",
        ),
        DeclareLaunchArgument(
            "use_tool_communication",
            default_value="false",
            description="Only available for e series!",
        ),
        DeclareLaunchArgument(
            "tool_parity",
            default_value="0",
            description="Parity communication for serial communication.",
        ),
        DeclareLaunchArgument(
            "tool_baud_rate",
            default_value="115200",
            description="Baud rate configuration for serial communication",
        ),
        DeclareLaunchArgument(
            "tool_stop_bits",
            default_value="1",
            description="Stop bits configuration for serial communication.",
        ),
        DeclareLaunchArgument(
            "tool_rx_idle_chars", 
            default_value="1.5",
            description="RX (Reciever) idle chars configuration for serial communication.",
        ),
        DeclareLaunchArgument(
            "tool_tx_idle_chars",
            default_value="3.5",
            description="TX (Transmitter) idle chars configuration for serial communication.",
        ),
        DeclareLaunchArgument(
            "tool_device_name",
            default_value="/tmp/ttyUR",
            description="File descriptor for the tool communication device.",
        ),
        DeclareLaunchArgument(
            "tool_tcp_port",
            default_value="54321",
            description="Remote port for the tool's serial device.",
        ),
        DeclareLaunchArgument(
            "tool_voltage",
            default_value="0",
            description="Tool voltage that will be setup.",
        ),
        DeclareLaunchArgument(
            "reverse_ip",
            default_value="172.22.22.10",
            description="IP for the robot controller to communicate back to the driver.",
        ),
        DeclareLaunchArgument(
            "script_command_port",
            default_value="50004",
            description="Port for forwarding URScript commands to the robot.",
        ),
        DeclareLaunchArgument(
            "reverse_port",
            default_value="50001",
            description="Port for sending cyclic instructions from the driver to the robot controller.",
        ),
        DeclareLaunchArgument(
            "script_sender_port",
            default_value="50002",
            description="Interface to query external_control URScript on this port.",
        ),
        DeclareLaunchArgument(
            "trajectory_port", 
            default_value="50003",
            description="Port for trajectory control", 
        ),
        DeclareLaunchArgument(
            "sim_ignition",
            default_value="false",
            description="Simulate with Ignition Gazebo",
        ),
        DeclareLaunchArgument(
            "sim_isaac",
            default_value="false",
            description="Simulate with Isaac Sim",
        ),
        DeclareLaunchArgument(
            "com_port",
            default_value="/dev/ttyUSB0",
            description="Communication port for the gripper",
        ),
    ]

    # Initializations
    ur_type = LaunchConfiguration("ur_type")
    robot_ip = LaunchConfiguration("robot_ip")
    safety_limits = LaunchConfiguration("safety_limits")
    safety_pos_margin = LaunchConfiguration("safety_pos_margin")
    safety_k_position = LaunchConfiguration("safety_k_position")
    runtime_config_package = LaunchConfiguration("runtime_config_package")
    controllers_file = LaunchConfiguration("controllers_file")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    moveit_config_package = LaunchConfiguration("moveit_config_package")
    moveit_config_file = LaunchConfiguration("moveit_config_file")
    moveit_joint_limits_file = LaunchConfiguration("moveit_joint_limits_file")
    kinematics_params_file = LaunchConfiguration("kinematics_params_file")
    tf_prefix = LaunchConfiguration("tf_prefix")
    use_fake_hardware = LaunchConfiguration("use_fake_hardware")
    fake_sensor_commands = LaunchConfiguration("fake_sensor_commands")
    controller_spawner_timeout = LaunchConfiguration("controller_spawner_timeout")
    launch_rviz = LaunchConfiguration("launch_rviz")
    headless_mode = LaunchConfiguration("headless_mode")
    launch_dashboard_client = LaunchConfiguration("launch_dashboard_client")
    use_tool_communication = LaunchConfiguration("use_tool_communication")
    tool_parity = LaunchConfiguration("tool_parity")
    tool_baud_rate = LaunchConfiguration("tool_baud_rate")
    tool_stop_bits = LaunchConfiguration("tool_stop_bits")
    tool_rx_idle_chars = LaunchConfiguration("tool_rx_idle_chars")
    tool_tx_idle_chars = LaunchConfiguration("tool_tx_idle_chars")
    tool_device_name = LaunchConfiguration("tool_device_name")
    tool_tcp_port = LaunchConfiguration("tool_tcp_port")
    tool_voltage = LaunchConfiguration("tool_voltage")
    reverse_ip = LaunchConfiguration("reverse_ip")
    script_command_port = LaunchConfiguration("script_command_port")
    reverse_port = LaunchConfiguration("reverse_port")
    script_sender_port = LaunchConfiguration("script_sender_port")
    trajectory_port = LaunchConfiguration("trajectory_port")
    sim_ignition = LaunchConfiguration("sim_ignition")
    sim_isaac = LaunchConfiguration("sim_isaac")
    com_port = LaunchConfiguration("com_port")

    # get these out of the way
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare(description_package), "rviz", "ur5e.rviz"]
    )

    controllers_file = PathJoinSubstitution(
        [FindPackageShare(runtime_config_package), "config", "controllers.yaml"]
    )

    # Combine the robot and gripper description commands into one
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare(description_package), "urdf", description_file]),
            " ",
            "safety_limits:=", safety_limits,
            " ",
            "safety_pos_margin:=", safety_pos_margin,
            " ",
            "safety_k_position:=", safety_k_position,
            " ",
            "name:=", "asbr",
            " ",
            "ur_type:=", ur_type,
            " ",
            "tf_prefix:=", tf_prefix,
            " ",
            "sim_ignition:=", sim_ignition,
            " ",
            "sim_isaac:=", sim_isaac,
            " ",
            "use_fake_hardware:=", use_fake_hardware,
            " ",
            "fake_sensor_commands:=", fake_sensor_commands,
            " ",
            "com_port:=", com_port,
            " ",
            "simulation_controllers:=", controllers_file
        ]
    )

    # Create the combined robot description
    robot_description = {"robot_description": robot_description_content}

    # MoveIt Configuration
    robot_description_semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare(moveit_config_package), "srdf", moveit_config_file]),
            " ",
            "name:=", "asbr",
            " ",
            #"prefix:=",
            #prefix,
            #" ",
        ]
    )
    
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    robot_description_kinematics_file = PathJoinSubstitution(
        [FindPackageShare(moveit_config_package), "config", kinematics_params_file]
    )
    robot_description_kinematics = ParameterFile(
        robot_description_kinematics_file,
        allow_substs=True,
    )

    
    # robot_description_planning = {
    #     "robot_description_planning": load_yaml("ur_description", "config/ur5e/joint_limits.yaml")
    # }

    # # Planning Configuration
    # # Planning Configuration
    # asbr_planning_pipeline_config = {
    #     "move_group": {
    #         "planning_plugin": "assignment3/ASBRPlanner",
    #         "start_state_max_bounds_error": 0.1,
    #     }
    # }
    # asbr_planning_yaml = load_yaml("assignment3", "config/asbr_planning.yaml")
    # asbr_planning_pipeline_config["move_group"].update(asbr_planning_yaml)

    robot_description_planning = {
        # load joint limits from the MoveIt config; returning None here breaks the launch
        "robot_description_planning": load_yaml("asbr_moveit", "config/joint_limits.yaml")
    }

    # Planning Configuration
    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
            "start_state_max_bounds_error": 0.1,
        }
    }

    ompl_planning_yaml = load_yaml("asbr_moveit","config/ompl_planning.yaml")
    ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)

    # Trajectory Execution Configuration
    moveit_controllers_yaml = load_yaml("asbr_moveit", "config/controllers.yaml")
    moveit_controllers = {
        "moveit_simple_controller_manager": moveit_controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    trajectory_execution = {
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
        "allow_trajectory_execution": True,
        "trajectory_execution.interrupt_execution": True,
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            #{"use_sim_time": use_sim_time},
        ],
    )

    # Controllers
    def controller_spawner(controllers, active=True):
        inactive_flags = ["--inactive"] if not active else []
        return Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "--controller-manager",
                "/controller_manager",
                "--controller-manager-timeout",
                controller_spawner_timeout,
            ]
            + inactive_flags
            + controllers,
        )

    controllers_active = [
        "ur5e_joint_state_broadcaster",
        "robotiq_joint_state_broadcaster",
        "ur5e_controller",
        "robotiq_controller"
    ]
    
    controllers_inactive = [
        "io_and_status_controller",
        "speed_scaling_state_broadcaster",
        "force_torque_sensor_broadcaster",
        "scaled_joint_trajectory_controller",
        "forward_velocity_controller",
        "forward_position_controller",
        #"robotiq_activation_controller"
    ]

    controller_spawners = [
        controller_spawner(controllers_active)] + [
        #controller_spawner(controllers_inactive, active=False)
    ]

    controller_manager = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            robot_description,
            ParameterFile(controllers_file, allow_substs=True),
        ],
        output="screen",
        condition=UnlessCondition(sim_ignition)
    )

    # rviz with moveit configuration
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare(moveit_config_package), "rviz", "view_robot.rviz"]
    )

    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_planning_pipeline_config,
            robot_description_kinematics,
            robot_description_planning,
        ],
    )


    nodes_to_start = [
        move_group_node,
        rviz_node,
        controller_manager,
    ] + controller_spawners

    return LaunchDescription(declared_arguments +  nodes_to_start)

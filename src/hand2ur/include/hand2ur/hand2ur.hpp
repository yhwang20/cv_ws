#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <memory>
#include <optional>
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <moveit_msgs/action/move_group.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <rclcpp/parameter_client.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <fstream>
#include <sstream>

class hand2ur : public rclcpp::Node{
private:
    // MoveIt action clients
    rclcpp_action::Client<moveit_msgs::action::MoveGroup>::SharedPtr move_group_client;
    std::optional<geometry_msgs::msg::Pose> last_sent_pose;

    // Configuration
    std::string arm_group_name {"ur5e"};
    std::string arm_end_effector_link {"tool0"};
    std::string arm_reference_frame {"base_link"};
    double arm_position_tolerance {0.01};
    double arm_orientation_tolerance {0.05};
    double arm_allowed_planning_time {1.0};
    int arm_planning_attempts {3};
    double arm_velocity_scale {0.5};
    double arm_accel_scale {0.5};
    double arm_ik_timeout {0.1};
    unsigned int arm_ik_attempts {5};
    double arm_joint_tolerance {0.005};
    double arm_ik_consistency_limit {0.05}; // radians
    double arm_joint_window {0.3};          // radians band around current state to limit deviation
    bool arm_use_joint_window {true};

    std::string gripper_group_name {"robotiq"};
    std::string gripper_joint_name {"robotiq_85_left_knuckle_joint"};
    double gripper_open_position {0.7};
    double gripper_close_position {0.0};
    double gripper_tolerance {0.01};

    std::shared_ptr<robot_model_loader::RobotModelLoader> robot_model_loader;
    std::unique_ptr<moveit::core::RobotState> robot_state;

    // Goals
    std::optional<geometry_msgs::msg::Pose> goal_pose;
    std::optional<std_msgs::msg::Bool> goal_pinch;

    // Subscriber for wrist pose
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr hand_subscriber;
    // Subscriber for pinch state
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr pinch_subscriber;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub;
    sensor_msgs::msg::JointState::SharedPtr latest_joint_state;

    // Timer for update rate
    std::shared_ptr<rclcpp::TimerBase> pose_timer;
    std::shared_ptr<rclcpp::TimerBase> pinch_timer;
    // TODO: Scaling parameters
public:
    // Constructor
    hand2ur( const std::string& name, rclcpp::NodeOptions options = rclcpp::NodeOptions() );
    // Initialize (set workspace limits? initial robot position? etc.)
    void initialize();
    // Subscriber callbacks
    void pose_callback( const geometry_msgs::msg::PoseStamped& pose );
    void pinch_callback( const std_msgs::msg::Bool& pinch );
    // Map helper function that applies coordinate scaling/transform
    geometry_msgs::msg::Pose coordmap( const geometry_msgs::msg::PoseStamped& hand_pose );
    // Helper functions to send MoveIt goals
    void updatePose();
    void updatePinch();
    bool poses_are_close(const geometry_msgs::msg::Pose& a,
                     const geometry_msgs::msg::Pose& b,
                     double pos_tol = 0.05, // 5 cm
                     double ang_tol = 0.1); // .1 rad
};

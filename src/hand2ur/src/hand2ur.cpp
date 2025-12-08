#include <hand2ur/hand2ur.hpp>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/kinematics_base/kinematics_base.h>
#include <rclcpp/parameter_map.hpp>

using namespace std::chrono_literals;
using MoveGroupGoalHandle = rclcpp_action::ClientGoalHandle<moveit_msgs::action::MoveGroup>;

hand2ur::hand2ur( const std::string& name, rclcpp::NodeOptions options )
    : Node(name, options.automatically_declare_parameters_from_overrides(true)) {

    std::cout << "Running hand2ur..." << std::endl;
    hand_subscriber = create_subscription<geometry_msgs::msg::PoseStamped>("wrist_pose", 10,
        std::bind(&hand2ur::pose_callback, this, std::placeholders::_1 ) );
    pinch_subscriber = create_subscription<std_msgs::msg::Bool>("is_pinching", 10,
        std::bind(&hand2ur::pinch_callback, this, std::placeholders::_1 ) );

    // for IK joint goal method
    joint_state_sub = create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10,
        [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
            latest_joint_state = msg;
    });

}

void hand2ur::initialize() {
    std::cout << "Initializing..." << std::endl;
    auto node_handle = shared_from_this();

    // client for move_action service
    move_group_client = rclcpp_action::create_client<moveit_msgs::action::MoveGroup>(node_handle, "/move_action");

    // params for IK solver (robot description, SRDF, kinematics solver)
    std::string robot_description_xml;
    std::string robot_description_semantic_xml;
    auto fetch_param = [this](const std::string& target_node, const std::string& param_name) -> std::optional<rclcpp::Parameter> {
        auto param_client = std::make_shared<rclcpp::SyncParametersClient>(this->shared_from_this(), target_node);
        if (!param_client->wait_for_service(2s)) {
            RCLCPP_WARN(this->get_logger(), "Param service for '%s' unavailable", target_node.c_str());
            return std::nullopt;
        }
        try {
            auto params = param_client->get_parameters({param_name});
            if (!params.empty()) {
                RCLCPP_INFO(this->get_logger(), "Fetched %s from '%s'", param_name.c_str(), target_node.c_str());
                return params.front();
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Failed to fetch %s from '%s': %s", param_name.c_str(), target_node.c_str(), e.what());
        }
        return std::nullopt;
    };
    auto set_or_declare = [this](const rclcpp::Parameter& p) -> bool {
        if (p.get_type() == rclcpp::ParameterType::PARAMETER_NOT_SET) {
            RCLCPP_WARN(this->get_logger(), "Parameter %s was not set on provider; skipping", p.get_name().c_str());
            return false;
        }
        if (!this->has_parameter(p.get_name())) {
            this->declare_parameter(p.get_name(), p.get_parameter_value());
        } else {
            this->set_parameter(p);
        }
        return true;
    };
    if (!this->has_parameter("robot_description")) {
        this->declare_parameter<std::string>("robot_description", "");
    }
    if (!this->has_parameter("robot_description_semantic")) {
        this->declare_parameter<std::string>("robot_description_semantic", "");
    }
    // get robot_description!
    for (const auto& provider : {std::string("robot_state_publisher"), std::string("move_group"), std::string("planning_scene_monitor")}) {
        if (robot_description_xml.empty()) {
            if (auto p = fetch_param(provider, "robot_description")) {
                robot_description_xml = p->as_string();
                set_or_declare(*p);
            }
        }
        if (robot_description_semantic_xml.empty()) {
            if (auto p = fetch_param(provider, "robot_description_semantic")) {
                robot_description_semantic_xml = p->as_string();
                set_or_declare(*p);
            }
        }
        if (!robot_description_xml.empty() && !robot_description_semantic_xml.empty()) {
            break;
        }
    }
    // manually get kinematics.yaml
    try {
        const auto pkg_share = ament_index_cpp::get_package_share_directory("asbr_moveit");
        const std::string kin_path = pkg_share + "/config/kinematics.yaml";
        auto param_map = rclcpp::parameter_map_from_yaml_file(kin_path);
        auto it = param_map.find("/**");
        if (it != param_map.end()) {
            for (const auto& p : it->second) {
                set_or_declare(p);
            }
            RCLCPP_INFO(this->get_logger(), "Loaded robot_description_kinematics from file: %s", kin_path.c_str());
        } else {
            RCLCPP_WARN(this->get_logger(), "No parameters found in kinematics.yaml at %s", kin_path.c_str());
        }
    } catch (const std::exception& e) {
        RCLCPP_WARN(this->get_logger(), "Failed to load kinematics.yaml from package: %s", e.what());
    }
    robot_model_loader = std::make_shared<robot_model_loader::RobotModelLoader>(node_handle, "robot_description");
    if (robot_model_loader->getModel()) {
        robot_state = std::make_unique<moveit::core::RobotState>(robot_model_loader->getModel());
        robot_state->setToDefaultValues();
    } else {
        RCLCPP_WARN(get_logger(), "Failed to load robot_description for IK seed; falling back to pose goals.");
    }

    if (!move_group_client->wait_for_action_server(5s)) {
        RCLCPP_ERROR(get_logger(), "MoveGroup action server not available");
    }

    // TIMERS for how often to send pose and pinch goals.
    pose_timer = create_wall_timer( 1000ms, std::bind(&hand2ur::updatePose, this ));
    pinch_timer = create_wall_timer( 1000ms, std::bind(&hand2ur::updatePinch, this ));
    std::cout << "==================" << std::endl;
    std::cout << "Done initializing." << std::endl;
    std::cout << "==================" << std::endl;
}

void hand2ur::pose_callback( const geometry_msgs::msg::PoseStamped& pose ) {
    this->goal_pose = coordmap(pose);
}

void hand2ur::pinch_callback( const std_msgs::msg::Bool& pinch ) {
    this->goal_pinch = pinch;
}

geometry_msgs::msg::Pose hand2ur::coordmap( const geometry_msgs::msg::PoseStamped& hand_pose ) {
    // TODO: Scale/transform the incoming hand pose into the robot frame.
    // For now, pass through the pose directly.
    return hand_pose.pose;
}

// Method for determining if current goal is close enough to
// previous goal (if so, don't move)
bool hand2ur::poses_are_close(const geometry_msgs::msg::Pose& a,
                              const geometry_msgs::msg::Pose& b,
                              double pos_tol,
                              double ang_tol)
{
    double dx = a.position.x - b.position.x;
    double dy = a.position.y - b.position.y;
    double dz = a.position.z - b.position.z;

    double pos_dist = std::sqrt(dx*dx + dy*dy + dz*dz);

    // Orientation difference via quaternion inner product
    double dot = a.orientation.x * b.orientation.x +
                 a.orientation.y * b.orientation.y +
                 a.orientation.z * b.orientation.z +
                 a.orientation.w * b.orientation.w;

    dot = std::fabs(dot);
    double ang_dist = 2.0 * std::acos(std::min(1.0, dot)); // radians

    return (pos_dist < pos_tol) && (ang_dist < ang_tol);
}


void hand2ur::updatePose() {
    if (!move_group_client || !move_group_client->action_server_is_ready()) {
        RCLCPP_WARN(get_logger(), "MoveGroup action server not ready");
        return;
    }

    if (this->goal_pose) {
        if (last_sent_pose && poses_are_close(this->goal_pose.value(), last_sent_pose.value())) {
            RCLCPP_INFO(get_logger(), "Pose unchanged, skipping MoveIt call.");
            return;
        }

        if (!latest_joint_state) {
            RCLCPP_WARN(get_logger(), "No joint_state received yet; skipping pose command.");
            return;
        }

        if (!robot_state) {
            RCLCPP_WARN(get_logger(), "RobotState not initialized; skipping pose command.");
            return;
        }

        // build goal
        moveit_msgs::action::MoveGroup::Goal goal;
        goal.request.group_name = arm_group_name;
        goal.request.allowed_planning_time = arm_allowed_planning_time;
        goal.request.num_planning_attempts = arm_planning_attempts;
        goal.request.max_velocity_scaling_factor = arm_velocity_scale;
        goal.request.max_acceleration_scaling_factor = arm_accel_scale;
        goal.planning_options.plan_only = false;
        goal.planning_options.look_around = false;
        goal.planning_options.replan = false;
        goal.planning_options.replan_attempts = 3;

        goal.request.start_state.joint_state = *latest_joint_state;
        goal.request.start_state.is_diff = true;

        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.frame_id = arm_reference_frame;
        pose_stamped.header.stamp = this->now();
        pose_stamped.pose = this->goal_pose.value();

        goal.request.goal_constraints.clear();

        const moveit::core::JointModelGroup* jmg = robot_state->getJointModelGroup(arm_group_name);
        if (!jmg) {
            RCLCPP_WARN(get_logger(), "Joint model group '%s' not found; cannot send IK joint goal.", arm_group_name.c_str());
            return;
        }

        robot_state->setVariableValues(*latest_joint_state);
        kinematics::KinematicsQueryOptions ik_options;
        ik_options.return_approximate_solution = false;

        bool ik_success = false;
        for (unsigned int attempt = 0; attempt < arm_ik_attempts && !ik_success; ++attempt) {
            ik_success = robot_state->setFromIK(
                jmg,
                this->goal_pose.value(),
                arm_end_effector_link,
                arm_ik_timeout,
                moveit::core::GroupStateValidityCallbackFn(),
                ik_options);
        }

        if (!ik_success) {
            RCLCPP_WARN(get_logger(), "IK failed for requested pose; skipping command to avoid large pose goal motion.");
            return;
        }

        auto joint_goal = kinematic_constraints::constructGoalConstraints(
            *robot_state, jmg, arm_joint_tolerance);
        goal.request.goal_constraints.push_back(joint_goal);
        RCLCPP_INFO(get_logger(), "Using IK-based joint goal for minimal motion.");

        // constrain motion to stay near the current joint state
        // IF arm_use_joint_window == true!
        if (arm_use_joint_window && arm_joint_window > 0.0) {
            std::unordered_map<std::string, double> current_positions;
            for (size_t i = 0; i < latest_joint_state->name.size(); ++i) {
                if (i < latest_joint_state->position.size()) {
                    current_positions[latest_joint_state->name[i]] = latest_joint_state->position[i];
                }
            }
            moveit_msgs::msg::Constraints joint_window_constraints;
            for (const auto& joint_name : jmg->getVariableNames()) {
                auto it = current_positions.find(joint_name);
                if (it == current_positions.end()) continue;
                moveit_msgs::msg::JointConstraint jc;
                jc.joint_name = joint_name;
                jc.position = it->second;
                jc.tolerance_above = arm_joint_window;
                jc.tolerance_below = arm_joint_window;
                jc.weight = 1.0;
                joint_window_constraints.joint_constraints.push_back(jc);
            }
            if (!joint_window_constraints.joint_constraints.empty()) {
                goal.request.path_constraints = joint_window_constraints;
                RCLCPP_INFO(get_logger(), "Applied joint window constraints (+/- %.3f rad).", arm_joint_window);
            }
        }


        rclcpp_action::Client<moveit_msgs::action::MoveGroup>::SendGoalOptions send_options;
        send_options.goal_response_callback = [this](MoveGroupGoalHandle::SharedPtr handle) {
            if (!handle) {
                RCLCPP_WARN(this->get_logger(), "MoveGroup goal rejected");
                return;
            }
        };
        send_options.result_callback = [this](const auto& result) {
            if (!result.result) {
                RCLCPP_WARN(this->get_logger(), "MoveGroup result missing");
                return;
            }
            if (result.result->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                RCLCPP_WARN(this->get_logger(), "MoveGroup execution failed with code %d", result.result->error_code.val);
            }
        };

        move_group_client->async_send_goal(goal, send_options);
        last_sent_pose = this->goal_pose.value();

        RCLCPP_INFO(get_logger(), "Sent pose command.");
    } else {
        RCLCPP_WARN(get_logger(), "Pose not found.");
    }
}

void hand2ur::updatePinch() {
    if (!move_group_client || !move_group_client->action_server_is_ready()) {
        RCLCPP_WARN(get_logger(), "MoveGroup action server not ready for gripper");
        return;
    }

    if (this->goal_pinch) {
        const bool should_close = goal_pinch.value().data;
        const std::string target_state = should_close ? "Close" : "Open";

        double target_position = should_close ? 0.8 : 0.0;   // MATCHES SRDF

        // Build a JointConstraint for the Robotiq gripper
        moveit_msgs::msg::JointConstraint jc;
        jc.joint_name = "robotiq_85_left_knuckle_joint";
        jc.position = target_position;
        jc.tolerance_above = 0.001;
        jc.tolerance_below = 0.001;
        jc.weight = 1.0;

        moveit_msgs::msg::Constraints c;
        c.joint_constraints.push_back(jc);

        moveit_msgs::action::MoveGroup::Goal goal;
        goal.request.group_name = gripper_group_name;
        goal.request.goal_constraints = {c};

        // Required MoveIt planning options
        goal.planning_options.plan_only = false;
        goal.planning_options.replan = false;
        goal.planning_options.look_around = false;

        rclcpp_action::Client<moveit_msgs::action::MoveGroup>::SendGoalOptions send_options;
        send_options.goal_response_callback = [this](MoveGroupGoalHandle::SharedPtr handle) {
            if (!handle) {
                RCLCPP_WARN(this->get_logger(), "Gripper MoveGroup goal rejected");
                return;
            }
        };
        send_options.result_callback = [this, target_state](const auto& result) {
            if (!result.result) {
                RCLCPP_WARN(this->get_logger(), "Gripper MoveGroup result missing");
                return;
            }
            if (result.result->error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
                RCLCPP_WARN(this->get_logger(), "Gripper MoveGroup action failed (code %d)", result.result->error_code.val);
            } else {
                RCLCPP_INFO(this->get_logger(), "Gripper state '%s' command complete", target_state.c_str());
            }
        };

        move_group_client->async_send_goal(goal, send_options);

        RCLCPP_INFO(get_logger(), "Sent gripper command: %s", target_state.c_str());    
    } else {
        RCLCPP_WARN(get_logger(), "Pinch not found.");
    }
}

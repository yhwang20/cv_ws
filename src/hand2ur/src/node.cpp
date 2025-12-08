#include<hand2ur/hand2ur.hpp>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    std::shared_ptr<hand2ur> node = std::make_shared<hand2ur>("hand2ur_node");
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

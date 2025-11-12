cmake_minimum_required(VERSION 3.8)
project(hand_landmarker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclpy REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PinchStatus.msg"
  "msg/HandPosition.msg"
  DEPENDENCIES std_msgs
)

# Install Python modules to local dist-packages
install(DIRECTORY hand_landmarker/
  DESTINATION local/lib/python3.10/dist-packages/hand_landmarker
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
  PATTERN "nodes" EXCLUDE
)

# Install Python executables (from nodes directory)
install(PROGRAMS
  hand_landmarker/nodes/hand_landmarker_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME hand_landmarker_node
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  FILES_MATCHING PATTERN "*.launch.py"
)

# Install model files
install(FILES
  hand_landmarker.task
  DESTINATION share/${PROJECT_NAME}/models
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()


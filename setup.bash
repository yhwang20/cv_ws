#!/bin/bash
# Custom workspace setup - properly sources all packages

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Source ROS 2
source /opt/ros/humble/setup.bash

# Add this workspace's install directories to AMENT_PREFIX_PATH  
export AMENT_PREFIX_PATH="${SCRIPT_DIR}/install/hand_landmarker:${SCRIPT_DIR}/install/camera_publisher_pkg:$AMENT_PREFIX_PATH"

# Source the workspace setup (sets CMAKE_PREFIX_PATH, PYTHONPATH, etc.)
source "${SCRIPT_DIR}/install/local_setup.bash" 2>/dev/null || true

echo "Workspace sourced successfully!"

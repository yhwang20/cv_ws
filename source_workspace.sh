#!/bin/bash
# Source ROS 2 base
source /opt/ros/humble/setup.bash

# Source workspace local setup
source /home/mariana/cv_ws/install/local_setup.bash

# Manually add workspace to AMENT_PREFIX_PATH if not already there
if [[ ":$AMENT_PREFIX_PATH:" != *":/home/mariana/cv_ws/install:"* ]]; then
    export AMENT_PREFIX_PATH="/home/mariana/cv_ws/install:$AMENT_PREFIX_PATH"
fi

echo "Workspace sourced. AMENT_PREFIX_PATH: $AMENT_PREFIX_PATH"


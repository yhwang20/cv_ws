#!/bin/bash
# Helper script to run the hand landmarker system

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Source the workspace
cd "$SCRIPT_DIR"
set +u  # Disable unbound variable check for compatibility
source setup.bash

# Launch the system
ros2 launch hand-landmarker hand_landmarker.launch.py


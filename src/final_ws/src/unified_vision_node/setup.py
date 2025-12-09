"""Setup script for unified_vision_node package."""

import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'unified_vision_node'

setup(
    name=package_name,
    version="0.1.0",
    author="Mariana",
    author_email="mariana@todo.todo",
    description="Unified ROS2 vision node integrating pinch and pose detection",
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.rviz')),
        ('share/' + package_name + '/weights', glob('weights/*')),
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    install_requires=[],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'unified_vision_node = unified_vision_node.nodes.unified_vision_node:main',
            'tf_publisher_node = unified_vision_node.nodes.tf_publisher_node:main',
            'pinching_gripper_node = unified_vision_node.nodes.pinching_gripper_node:main',
        ],
    },
)


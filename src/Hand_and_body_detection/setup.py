"""Setup script for hand_body_detection package."""

import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'hand_body_detection'

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name=package_name,
    version="0.1.0",
    author="Mariana",
    author_email="mariana@todo.todo",
    description="A ROS2 package for real-time hand and body pose detection using MediaPipe",
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/data', glob('data/*.jpg')),
        ('share/' + package_name + '/data', glob('data/*.png')),
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'pose_detector_node = hand_body_detection.nodes.pose_detector_node:main',
            'pose_image_test = hand_body_detection.scripts.hand_shoulder_image:main',
            'pose_video_test = hand_body_detection.scripts.hand_shoulder_video:main',
        ],
    },
)


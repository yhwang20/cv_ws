"""Setup script for hand_landmarker package."""

import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'hand_landmarker'

# Note: Custom messages are handled by CMake/ament_cmake, not setup.py

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name=package_name,
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for real-time hand landmark detection using MediaPipe",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hand-landmarker",
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob('*.task')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
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
            'hand_landmarker_node = hand_landmarker.nodes.hand_landmarker_node:main',
        ],
    },
)


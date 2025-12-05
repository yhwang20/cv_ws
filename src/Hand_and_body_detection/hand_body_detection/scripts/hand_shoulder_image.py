#!/usr/bin/env python3
"""Test script for hand and shoulder detection in static images."""

import os
import sys
import cv2
import numpy as np

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pose_detector import PoseDetector, calculate_angle


def main():
    """Run pose detection on static images."""
    
    # Get images from data folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(os.path.dirname(script_dir))
    
    IMAGE_FILES = [
        os.path.join(package_dir, "data", "demo.jpg"),
        os.path.join(package_dir, "data", "testing2.jpg")
    ]
    
    # Create output folder if it doesn't exist
    OUTPUT_DIR = os.path.join(os.path.dirname(script_dir), "src", "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    BG_COLOR = (192, 192, 192)  # gray
    
    # Initialize detector
    detector = PoseDetector(min_detection_confidence=0.5, model_complexity=2)
    
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        if image is None:
            print(f"ERROR: Cannot read image: {file}")
            continue
        else:
            print(f"Loaded image: {file} with shape {image.shape}")
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = detector.detect(rgb_image)
        
        if not results.pose_landmarks:
            print(f"No pose detected in: {file}")
            continue
        else:
            print(f"Pose detected! {len(results.pose_landmarks.landmark)} landmarks found.")
        
        annotated_image = rgb_image.copy()
        
        # Apply background segmentation if available
        if results.segmentation_mask is not None:
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
        
        # Get right arm positions and angles
        positions = detector.get_wrist_shoulder_positions(results, 'RIGHT')
        
        if positions:
            shoulder = positions['shoulder']
            elbow = positions['elbow']
            wrist = positions['wrist']
            
            # Get hip for angle calculation
            landmarks = results.pose_landmarks.landmark
            right_hip = landmarks[detector.detector.mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_hip_pos = (right_hip.x, right_hip.y, right_hip.z)
            
            # Calculate angles
            base_angle = calculate_angle(right_hip_pos, shoulder, elbow)
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            
            # Convert to pixel coordinates for text
            height, width = image.shape[:2]
            shoulder_pixel = (int(shoulder[0] * width), int(shoulder[1] * height))
            elbow_pixel = (int(elbow[0] * width), int(elbow[1] * height))
            
            # Add angle text
            cv2.putText(annotated_image, str(int(base_angle)),
                       shoulder_pixel,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated_image, str(int(elbow_angle)),
                       elbow_pixel,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            print(f"Shoulder Coordinate of image {idx + 1}:", shoulder)
            print(f"Elbow Coordinate of image {idx + 1}:", elbow)
            print(f"Wrist Coordinate of image {idx + 1}:", wrist)
        
        # Draw landmarks
        detector.draw_landmarks(annotated_image, results)
        
        # Convert back to BGR and save
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(OUTPUT_DIR, f"annotated_{idx}.png")
        cv2.imwrite(out_path, annotated_bgr)
        print(f"Saved annotated image: {out_path}")
    
    detector.close()
    print("Processing complete!")


if __name__ == '__main__':
    main()


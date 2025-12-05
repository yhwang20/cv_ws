#!/usr/bin/env python3
"""Test script for hand and shoulder detection in video feed."""

import sys
import os
import cv2
import numpy as np

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pose_detector import PoseDetector, calculate_angle


def main():
    """Run pose detection on video feed."""
    
    # Initialize detector
    detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Video capture
    cap = cv2.VideoCapture(0)
    
    print("Starting video capture. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = detector.detect(rgb_image)
        
        # Create annotated image
        annotated_image = rgb_image.copy()
        
        try:
            # Get right arm positions
            positions = detector.get_wrist_shoulder_positions(results, 'RIGHT')
            
            if positions:
                shoulder = positions['shoulder']
                elbow = positions['elbow']
                wrist = positions['wrist']
                
                # Get hip for angle calculation
                landmarks = results.pose_landmarks.landmark
                right_hip = landmarks[detector.mp_pose.PoseLandmark.RIGHT_HIP.value]
                right_hip_pos = (right_hip.x, right_hip.y, right_hip.z)
                
                # Calculate angles
                base_angle = calculate_angle(right_hip_pos, shoulder, elbow)
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                
                # Convert to pixel coordinates for display
                height, width = frame.shape[:2]
                shoulder_pixel = (int(shoulder[0] * width), int(shoulder[1] * height))
                elbow_pixel = (int(elbow[0] * width), int(elbow[1] * height))
                
                # Add angle text
                cv2.putText(annotated_image, str(int(base_angle)),
                           shoulder_pixel,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_image, str(int(elbow_angle)),
                           elbow_pixel,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception as e:
            pass  # Continue even if angle calculation fails
        
        # Draw landmarks
        detector.draw_landmarks(annotated_image, results)
        
        # Convert back to BGR for display
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        # Display
        cv2.imshow('MediaPipe Pose Detection', annotated_bgr)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Video capture ended.")


if __name__ == '__main__':
    main()


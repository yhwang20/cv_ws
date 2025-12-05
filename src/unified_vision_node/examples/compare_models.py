#!/usr/bin/env python3
"""
example comparing mediapipe vs custom cnn pinch detection models
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from unified_vision_node.models.mediapipe_finger_pinch import MediapipeFingerPinch
from unified_vision_node.models.custom_finger_pinch import CustomFingerPinch


def main():
    # initialize both models
    print("initializing models...")
    mediapipe_model = MediapipeFingerPinch(
        model_path='hand_landmarker.task',
        pinch_threshold=0.65
    )
    
    custom_model = CustomFingerPinch(
        model_path='hand_gesture_model.pth',
        threshold=0.5
    )
    
    # open webcam
    cap = cv2.VideoCapture(0)
    
    print("press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # convert to rgb
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # run both models
        mp_result = mediapipe_model(rgb_frame, None, None)
        custom_result = custom_model(rgb_frame, None, None)
        
        # create side-by-side comparison
        mp_annotated = mp_result['annotated_image']
        custom_annotated = custom_result['annotated_image']
        
        # convert back to bgr for display
        mp_bgr = cv2.cvtColor(mp_annotated, cv2.COLOR_RGB2BGR)
        custom_bgr = cv2.cvtColor(custom_annotated, cv2.COLOR_RGB2BGR)
        
        # add labels
        cv2.putText(mp_bgr, "MediaPipe", (10, mp_bgr.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(custom_bgr, "Custom CNN", (10, custom_bgr.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # stack horizontally
        comparison = cv2.hconcat([mp_bgr, custom_bgr])
        
        # display
        cv2.imshow('Model Comparison: MediaPipe (left) vs Custom CNN (right)', comparison)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


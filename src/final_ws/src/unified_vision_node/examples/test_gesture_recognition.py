#!/usr/bin/env python3
"""
Example script to test MediaPipe Gesture Recognition model.
This script demonstrates how to use the gesture recognizer with a webcam or image.
"""

import cv2
import numpy as np
import sys
import os

# Add the parent directory to the path to import the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unified_vision_node.models.mediapipe_gesture_recognition import MediapipeGestureRecognition


def test_with_webcam(model_path='gesture_recognizer.task'):
    """Test gesture recognition with webcam feed."""
    
    # Initialize the model
    print("Initializing MediaPipe Gesture Recognition...")
    try:
        recognizer = MediapipeGestureRecognition(
            model_path=model_path,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            gesture_classes=['Closed_Fist', 'Open_Palm'],
            num_hands=2
        )
        print("Model initialized successfully!")
        print(f"Gesture classes: {recognizer.gesture_classes}")
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("\nMake sure you have downloaded the gesture_recognizer.task model file.")
        print("Download it from: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task")
        return
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("\nStarting webcam feed...")
    print("Press 'q' to quit")
    print("\nTry making these gestures:")
    print("  - Open Palm (hand open with fingers spread)")
    print("  - Closed Fist (hand closed)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run gesture recognition
        result = recognizer(rgb_frame, depth_image=None, camera_info=None)
        
        # Convert annotated image back to BGR for display
        annotated_frame = cv2.cvtColor(result['annotated_image'], cv2.COLOR_RGB2BGR)
        
        # Add one-hot encoding display
        one_hot_text = "One-hot: [" + ", ".join([f"{v:.1f}" for v in result['gesture_one_hot']]) + "]"
        cv2.putText(
            annotated_frame,
            one_hot_text,
            (10, annotated_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Display the frame
        cv2.imshow('MediaPipe Gesture Recognition', annotated_frame)
        
        # Print results to console
        if result['gesture_name']:
            print(f"\rGesture: {result['gesture_name']:<15} "
                  f"Confidence: {result['gesture_confidence']:.2f}  "
                  f"Hand: {result['handedness']:<10}  "
                  f"One-hot: {result['gesture_one_hot']}", end='', flush=True)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n\nWebcam test completed!")


def test_with_image(image_path, model_path='gesture_recognizer.task'):
    """Test gesture recognition with a single image."""
    
    # Initialize the model
    print("Initializing MediaPipe Gesture Recognition...")
    try:
        recognizer = MediapipeGestureRecognition(
            model_path=model_path,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            gesture_classes=['Closed_Fist', 'Open_Palm'],
            num_hands=2
        )
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        return
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run gesture recognition
    result = recognizer(rgb_image, depth_image=None, camera_info=None)
    
    # Display results
    print("\n=== Gesture Recognition Results ===")
    print(f"Gesture Name: {result['gesture_name']}")
    print(f"Confidence: {result['gesture_confidence']:.4f}")
    print(f"Handedness: {result['handedness']}")
    print(f"One-hot encoding: {result['gesture_one_hot']}")
    print(f"Gesture classes: {recognizer.gesture_classes}")
    
    # Convert annotated image back to BGR for display
    annotated_image = cv2.cvtColor(result['annotated_image'], cv2.COLOR_RGB2BGR)
    
    # Display the image
    cv2.imshow('MediaPipe Gesture Recognition', annotated_image)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MediaPipe Gesture Recognition')
    parser.add_argument('--mode', type=str, default='webcam', choices=['webcam', 'image'],
                        help='Test mode: webcam or image')
    parser.add_argument('--image', type=str, help='Path to image file (required if mode=image)')
    parser.add_argument('--model', type=str, default='gesture_recognizer.task',
                        help='Path to gesture recognizer model file')
    
    args = parser.parse_args()
    
    if args.mode == 'webcam':
        test_with_webcam(model_path=args.model)
    elif args.mode == 'image':
        if not args.image:
            print("Error: --image argument required for image mode")
            sys.exit(1)
        test_with_image(args.image, model_path=args.model)


"""Basic example: Detect hand landmarks in a single image."""

import cv2
import sys
import os

# Add parent directory to path to import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hand_landmarker import HandLandmarkDetector, draw_landmarks_on_image


def main():
    """Main function to demonstrate basic hand landmark detection."""
    
    # Configuration
    # Look for model file in current directory or parent directory
    model_path = "hand_landmarker.task"  # Path to the model file
    if not os.path.exists(model_path):
        parent_model = os.path.join(os.path.dirname(__file__), "..", "hand_landmarker.task")
        if os.path.exists(parent_model):
            model_path = parent_model
    
    image_path = "image_pinch.jpg"  # Path to input image
    if not os.path.exists(image_path):
        parent_image = os.path.join(os.path.dirname(__file__), "..", "image_pinch.jpg")
        if os.path.exists(parent_image):
            image_path = parent_image
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("Please download it using:")
        print("wget https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return
    
    # Check if image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        print("Please provide a valid image path.")
        return
    
    print("Initializing hand landmark detector...")
    # Initialize detector
    detector = HandLandmarkDetector(model_path)
    
    print(f"Loading image from '{image_path}'...")
    # Load image using OpenCV (BGR format)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from '{image_path}'")
        return
    
    print("Detecting hand landmarks...")
    # Detect hand landmarks
    result = detector.detect(image)
    
    # Print detection results
    num_hands = len(result.hand_landmarks)
    print(f"Detected {num_hands} hand(s)")
    
    for idx, (hand_landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
        print(f"  Hand {idx + 1}: {handedness[0].category_name} (confidence: {handedness[0].score:.2f})")
        print(f"    Number of landmarks: {len(hand_landmarks)}")
    
    print("Drawing landmarks on image...")
    # Convert BGR to RGB for visualization
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw landmarks
    annotated_image = draw_landmarks_on_image(rgb_image, result)
    
    # Convert back to BGR for OpenCV display
    annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    
    # Save output
    output_path = "output_annotated.jpg"
    cv2.imwrite(output_path, annotated_bgr)
    print(f"Annotated image saved to '{output_path}'")
    
    # Display image
    print("Displaying image (press any key to close)...")
    cv2.imshow("Hand Landmarks Detection", annotated_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Done!")


if __name__ == "__main__":
    main()


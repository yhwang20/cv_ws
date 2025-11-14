import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque

class LiveGesturePredictor:
    """Real-time hand gesture prediction using trained CNN."""
    
    def __init__(self, model_path='hand_gesture_model.keras', img_size=(128, 128)):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model
            img_size: Input image size for model
        """
        self.img_size = img_size
        self.class_names = ['UNPINCHED', 'PINCHED']
        self.model = None
        self.prediction_history = deque(maxlen=10)  # Smooth predictions
        
        # Load model
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for prediction.
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Preprocessed frame ready for model
        """
        # Resize to model input size
        img = cv2.resize(frame, self.img_size)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize
        img = img.astype(np.float32) / 255.0
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return img
    
    def predict(self, frame):
        """
        Make prediction on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (class_name, confidence, raw_prediction)
        """
        preprocessed = self.preprocess_frame(frame)
        prediction = self.model.predict(preprocessed, verbose=0)[0][0]
        
        # Add to history for smoothing
        self.prediction_history.append(prediction)
        
        # Use average of recent predictions for stability
        smoothed_prediction = np.mean(self.prediction_history)
        
        # Determine class
        class_idx = 1 if smoothed_prediction > 0.5 else 0
        confidence = smoothed_prediction if class_idx == 1 else 1 - smoothed_prediction
        
        return self.class_names[class_idx], confidence, smoothed_prediction
    
    def draw_prediction_ui(self, frame, prediction, confidence, raw_value):
        """
        Draw prediction information on frame.
        
        Args:
            frame: Frame to draw on
            prediction: Predicted class name
            confidence: Confidence value (0-1)
            raw_value: Raw prediction value
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for info panel
        overlay = frame.copy()
        
        # Draw background panel
        cv2.rectangle(overlay, (10, 10), (w - 10, 200), (0, 0, 0), -1)
        frame_with_overlay = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Draw prediction text
        color = (0, 255, 0) if prediction == 'UNPINCHED' else (0, 165, 255)
        cv2.putText(frame_with_overlay, f"Prediction: {prediction}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw confidence
        conf_text = f"Confidence: {confidence * 100:.1f}%"
        cv2.putText(frame_with_overlay, conf_text, 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw confidence bar
        bar_width = int((w - 60) * confidence)
        cv2.rectangle(frame_with_overlay, (20, 110), (w - 40, 140), (100, 100, 100), -1)
        cv2.rectangle(frame_with_overlay, (20, 110), (20 + bar_width, 140), color, -1)
        cv2.rectangle(frame_with_overlay, (20, 110), (w - 40, 140), (255, 255, 255), 2)
        
        # Draw raw value indicator
        cv2.putText(frame_with_overlay, f"Raw value: {raw_value:.3f}", 
                   (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw threshold line
        threshold_x = 20 + int((w - 60) * 0.5)
        cv2.line(frame_with_overlay, (threshold_x, 110), (threshold_x, 140), 
                (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(frame_with_overlay, "Press 'q' to quit", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw FPS counter region
        cv2.putText(frame_with_overlay, "Live Prediction", 
                   (w - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame_with_overlay
    
    def run(self):
        """Run live prediction loop."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("=" * 60)
        print("Live Hand Gesture Prediction")
        print("=" * 60)
        print("Starting camera... Press 'q' to quit")
        print("=" * 60)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Make prediction every frame for smooth updates
            prediction, confidence, raw_value = self.predict(frame)
            
            # Draw UI
            display_frame = self.draw_prediction_ui(frame, prediction, confidence, raw_value)
            
            # Show frame
            cv2.imshow('Hand Gesture Prediction', display_frame)
            
            # Print prediction occasionally
            frame_count += 1
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"Prediction: {prediction} | Confidence: {confidence * 100:.1f}%")
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("Prediction session ended")
        print("=" * 60)


def main():
    """Run live prediction."""
    try:
        predictor = LiveGesturePredictor(model_path='hand_gesture_model.keras')
        predictor.run()
    except FileNotFoundError:
        print("\nError: Model file 'hand_gesture_model.keras' not found.")
        print("Please train the model first using train_cnn.py")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
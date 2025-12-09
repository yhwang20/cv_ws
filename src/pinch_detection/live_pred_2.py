import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from collections import deque
import mediapipe as mp

class LiveGesturePredictor:
    """Real-time hand gesture prediction using trained CNN (PyTorch) with MediaPipe hand cropping."""
    
    def __init__(self, model_path='hand_gesture_model.pth', img_size=(224, 224), device=None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model
            img_size: Input image size for model
            device: Device to use (cuda/cpu)
        """
        self.img_size = img_size
        self.class_names = ['UNPINCHED', 'PINCHED']
        self.model = None
        self.prediction_history = deque(maxlen=10)  # Smooth predictions
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        # Load model
        self.load_model(model_path)
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # For hand detection feedback
        self.hand_detected = False
        self.hand_bbox = None
    
    def load_model(self, model_path):
        """Load the trained PyTorch model."""
        try:
            # Build model architecture
            base_model = models.efficientnet_b0(weights=None)
            num_features = base_model.classifier[1].in_features
            
            base_model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            base_model.load_state_dict(checkpoint['model_state_dict'])
            base_model.eval()
            
            self.model = base_model.to(self.device)
            
            # Get image size from checkpoint if available
            if 'img_size' in checkpoint:
                self.img_size = checkpoint['img_size']
            
            print(f"Model loaded successfully from {model_path}")
            print(f"Using input size: {self.img_size}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def crop_hand_region(self, frame):
        """
        Detect and crop hand region using MediaPipe.
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Cropped hand image or None if no hand detected
        """
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            self.hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get image dimensions
            h, w, _ = frame.shape
            
            # Extract landmark coordinates
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            # Calculate bounding box with padding
            x_min = int(min(x_coords))
            x_max = int(max(x_coords))
            y_min = int(min(y_coords))
            y_max = int(max(y_coords))
            
            # Add padding (20% on each side)
            padding_x = int((x_max - x_min) * 0.2)
            padding_y = int((y_max - y_min) * 0.2)
            
            x_min = max(0, x_min - padding_x)
            x_max = min(w, x_max + padding_x)
            y_min = max(0, y_min - padding_y)
            y_max = min(h, y_max + padding_y)
            
            # Store bbox for visualization
            self.hand_bbox = (x_min, y_min, x_max, y_max)
            
            # Crop hand region
            hand_crop = frame[y_min:y_max, x_min:x_max]
            
            return hand_crop, hand_landmarks
        else:
            self.hand_detected = False
            self.hand_bbox = None
            return None, None
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for prediction.
        
        Args:
            frame: Input frame (cropped hand region)
            
        Returns:
            Preprocessed frame ready for model
        """
        # Resize to model input size
        img = cv2.resize(frame, self.img_size)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms (normalization)
        img = self.transform(img)
        
        # Add batch dimension
        img = img.unsqueeze(0)
        
        return img
    
    def predict(self, frame):
        """
        Make prediction on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (class_name, confidence, raw_prediction, hand_detected)
        """
        # Crop to hand region
        hand_crop, hand_landmarks = self.crop_hand_region(frame)
        
        if hand_crop is None or hand_crop.size == 0:
            # No hand detected
            return None, 0.0, 0.0, False
        
        # Preprocess and predict
        preprocessed = self.preprocess_frame(hand_crop)
        preprocessed = preprocessed.to(self.device)
        
        with torch.no_grad():
            prediction = self.model(preprocessed).cpu().numpy()[0][0]
        
        # Add to history for smoothing
        self.prediction_history.append(prediction)
        
        # Use average of recent predictions for stability
        smoothed_prediction = np.mean(self.prediction_history)
        
        # Determine class
        class_idx = 1 if smoothed_prediction > 0.5 else 0
        confidence = smoothed_prediction if class_idx == 1 else 1 - smoothed_prediction
        
        return self.class_names[class_idx], confidence, smoothed_prediction, True
    
    def draw_prediction_ui(self, frame, prediction, confidence, raw_value, hand_detected):
        """
        Draw prediction information on frame.
        
        Args:
            frame: Frame to draw on
            prediction: Predicted class name
            confidence: Confidence value (0-1)
            raw_value: Raw prediction value
            hand_detected: Whether hand was detected
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for info panel
        overlay = frame.copy()
        
        # Draw hand bounding box if detected
        if hand_detected and self.hand_bbox is not None:
            x_min, y_min, x_max, y_max = self.hand_bbox
            color = (0, 255, 0) if prediction == 'UNPINCHED' else (0, 165, 255)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 3)
            cv2.putText(overlay, "Hand Detected", (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw background panel
        panel_height = 220 if hand_detected else 120
        cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
        frame_with_overlay = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        if hand_detected:
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
            
            # MediaPipe indicator
            cv2.putText(frame_with_overlay, "MediaPipe + PyTorch", 
                       (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        else:
            # No hand detected message
            cv2.putText(frame_with_overlay, "No hand detected", 
                       (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame_with_overlay, "Show your hand to the camera", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(frame_with_overlay, "Press 'q' to quit", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return frame_with_overlay
    
    def run(self, show_fps=True):
        """
        Run live prediction loop.
        
        Args:
            show_fps: Whether to display FPS counter
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("=" * 60)
        print("Live Hand Gesture Prediction (MediaPipe + PyTorch)")
        print("=" * 60)
        print(f"Model input size: {self.img_size}")
        print("Hand detection: MediaPipe Hands")
        print("Starting camera... Press 'q' to quit")
        print("=" * 60)
        
        frame_count = 0
        import time
        fps_start_time = time.time()
        fps = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Make prediction (includes hand detection and cropping)
            prediction, confidence, raw_value, hand_detected = self.predict(frame)
            
            # Draw UI
            display_frame = self.draw_prediction_ui(frame, prediction, confidence, 
                                                   raw_value, hand_detected)
            
            # Calculate and display FPS
            if show_fps:
                frame_count += 1
                if frame_count % 10 == 0:
                    fps_end_time = time.time()
                    fps = 10 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                
                cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                           (display_frame.shape[1] - 150, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            # Show frame
            cv2.imshow('Hand Gesture Prediction', display_frame)
            
            # Print prediction occasionally
            if frame_count % 30 == 0 and hand_detected:
                print(f"Prediction: {prediction} | Confidence: {confidence * 100:.1f}%")
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        
        print("\n" + "=" * 60)
        print("Prediction session ended")
        print("=" * 60)


def main():
    """Run live prediction."""
    try:
        predictor = LiveGesturePredictor(
            # model_path='proposed_hand_gesture_model.pth',
            model_path='hand_gesture_model_v02.pth',
            img_size=(224, 224)  # Default for EfficientNet
        )
        predictor.run(show_fps=True)
    except FileNotFoundError:
        print("\nError: Model file 'proposed_hand_gesture_model.pth' not found.")
        print("Please train the model first using the PyTorch training script")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
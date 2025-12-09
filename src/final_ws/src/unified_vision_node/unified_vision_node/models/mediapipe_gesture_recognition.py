import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


# MediaPipe Gesture Recognition model out-of-the-box (can classify open-palm, closed-fist, etc.)

class MediapipeGestureRecognition:
    
    def __init__(self, model_path='gesture_recognizer.task',
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 gesture_classes=['Closed_Fist', 'Open_Palm'],
                 num_hands=2):
        """
        Initialize MediaPipe Gesture Recognizer.
        
        Args:
            model_path: Path to the gesture recognizer model file (.task)
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
            gesture_classes: List of gesture class names to recognize (one-hot encoded output)
            num_hands: Maximum number of hands to detect
        """
        
        # Store gesture classes for one-hot encoding
        self.gesture_classes = gesture_classes
        self.num_hands = num_hands
        
        # Create MediaPipe gesture recognizer
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        
    def __call__(self, rgb_image, depth_image, camera_info):
        """
        Process an image and recognize hand gestures.
        
        Args:
            rgb_image: RGB image as numpy array
            depth_image: Depth image (not used in this model)
            camera_info: Camera intrinsic parameters (not used in this model)
            
        Returns:
            Dictionary containing:
                - 'is_pinching': bool - True if Closed_Fist detected, False otherwise
                - 'gesture_one_hot': One-hot encoded vector of detected gestures
                - 'gesture_name': Name of the detected gesture (or None)
                - 'gesture_confidence': Confidence score of the gesture
                - 'handedness': Left/Right hand classification
                - 'hand_landmarks': Detected hand landmarks
                - 'palm_centroid_pixel': (x, y) pixel coordinates of palm center
                - 'normalized_spread': Gesture confidence score (0.0 to 1.0)
                - 'annotated_image': Image with drawn landmarks and labels
        """
        # Convert to MediaPipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self.recognizer.recognize(mp_image)
        
        # Initialize output with same interface as other pinch models
        output = {
            'is_pinching': False,  # True if Closed_Fist detected
            'gesture_one_hot': np.zeros(len(self.gesture_classes), dtype=np.float32),
            'gesture_name': None,
            'gesture_confidence': 0.0,
            'handedness': None,
            'hand_landmarks': result.hand_landmarks,
            'palm_centroid_pixel': None,
            'normalized_spread': 0.0,
            'annotated_image': rgb_image.copy()
        }
        
        # Draw landmarks on image
        output['annotated_image'] = self._draw_landmarks_on_image(rgb_image, result)
        
        # Process detected gestures
        if len(result.gestures) > 0 and len(result.gestures[0]) > 0:
            # Take the first detected hand's top gesture
            top_gesture = result.gestures[0][0]
            gesture_name = top_gesture.category_name
            gesture_confidence = top_gesture.score
            
            output['gesture_name'] = gesture_name
            output['gesture_confidence'] = float(gesture_confidence)
            output['normalized_spread'] = float(gesture_confidence)
            
            # Map gesture to is_pinching boolean
            # Closed_Fist = pinching (True), Open_Palm = not pinching (False)
            if gesture_name == 'Closed_Fist':
                output['is_pinching'] = True
            elif gesture_name == 'Open_Palm':
                output['is_pinching'] = False
            
            # Create one-hot encoding
            if gesture_name in self.gesture_classes:
                gesture_idx = self.gesture_classes.index(gesture_name)
                output['gesture_one_hot'][gesture_idx] = 1.0
            
            # Calculate palm centroid from landmarks
            if len(result.hand_landmarks) > 0:
                palm_x, palm_y = self._calculate_palm_centroid(result.hand_landmarks[0], rgb_image.shape)
                output['palm_centroid_pixel'] = (palm_x, palm_y)
                
                # Draw palm centroid marker
                cv2.circle(output['annotated_image'], (palm_x, palm_y), 10, (0, 0, 255), -1)
                cv2.circle(output['annotated_image'], (palm_x, palm_y), 8, (255, 255, 255), 2)
            
            # Get handedness
            if len(result.handedness) > 0 and len(result.handedness[0]) > 0:
                output['handedness'] = result.handedness[0][0].category_name
            
            # Add gesture label to image
            height, width = rgb_image.shape[:2]
            label_text = f"{gesture_name}: {gesture_confidence:.2f}"
            cv2.putText(
                output['annotated_image'],
                label_text,
                (10, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
            
            # Add handedness label if available
            if output['handedness']:
                cv2.putText(
                    output['annotated_image'],
                    f"Hand: {output['handedness']}",
                    (10, 80),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.0,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA
                )
        
        return output
    
    def _draw_landmarks_on_image(self, rgb_image, detection_result):
        """Draw hand landmarks and gesture labels on the image."""
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        gestures_list = detection_result.gestures
        annotated_image = np.copy(rgb_image)
        
        # Draw landmarks for each detected hand
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Convert landmarks to protobuf format
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                ) for landmark in hand_landmarks
            ])
            
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
            
            # Get gesture and handedness for this hand
            if idx < len(gestures_list) and len(gestures_list[idx]) > 0:
                gesture = gestures_list[idx][0]
                gesture_text = f"{gesture.category_name} ({gesture.score:.2f})"
            else:
                gesture_text = "No gesture"
            
            if idx < len(handedness_list) and len(handedness_list[idx]) > 0:
                handedness = handedness_list[idx][0].category_name
            else:
                handedness = "Unknown"
            
            # Calculate position for label (near the hand)
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - 30
            
            # Draw handedness label
            cv2.putText(
                annotated_image,
                f"{handedness}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (88, 205, 54),
                2,
                cv2.LINE_AA
            )
            
            # Draw gesture label
            cv2.putText(
                annotated_image,
                gesture_text,
                (text_x, text_y + 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (255, 165, 0),
                2,
                cv2.LINE_AA
            )
        
        return annotated_image
    
    def get_gesture_vector(self):
        """Return the list of gesture classes for reference."""
        return self.gesture_classes
    
    def _calculate_palm_centroid(self, hand_landmarks, image_shape):
        """
        Calculate palm centroid from hand landmarks.
        Uses the base of each finger (MCP joints) plus wrist.
        """
        # Palm landmark indices: wrist, thumb MCP, index MCP, middle MCP, ring MCP, pinky MCP
        palm_indices = [0, 5, 9, 13, 17]
        
        x_norm = np.mean([hand_landmarks[i].x for i in palm_indices])
        y_norm = np.mean([hand_landmarks[i].y for i in palm_indices])
        
        height, width = image_shape[:2]
        x_pixel = int(x_norm * width)
        y_pixel = int(y_norm * height)
        
        return x_pixel, y_pixel


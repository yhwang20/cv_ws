import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

class CustomMediapipeGestureRecognizer:
    """
    Custom wrapper around MediaPipe Gesture Recognition model.
    Provides a clean interface compatible with ModelComparison.
    """

    def __init__(self, model_path: str,
                 gesture_classes=['Closed_Fist', 'Open_Palm'],
                 num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        import os
        self.gesture_classes = gesture_classes
        self.num_hands = num_hands

        # Initialize MediaPipe Tasks gesture recognizer
        # MediaPipe has issues with absolute paths on Windows, so read the model file
        # and pass it as a buffer instead
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_content = f.read()
            base_options = python.BaseOptions(model_asset_buffer=model_content)
        else:
            # Fallback to direct path if file doesn't exist
            base_options = python.BaseOptions(model_asset_path=model_path)
        
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.mp_hands = mp.solutions.hands
        self.drawing_utils = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def __call__(self, rgb_image: np.ndarray):
        """
        Process a single RGB image and return structured gesture info.

        Returns:
            dict with keys:
                'is_pinching', 'gesture_name', 'gesture_confidence',
                'gesture_one_hot', 'hand_landmarks', 'handedness',
                'palm_centroid_pixel', 'normalized_spread', 'annotated_image'
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self.recognizer.recognize(mp_image)

        # Initialize default output
        output = {
            'is_pinching': False,
            'gesture_name': None,
            'gesture_confidence': 0.0,
            'gesture_one_hot': np.zeros(len(self.gesture_classes), dtype=np.float32),
            'hand_landmarks': [],
            'handedness': None,
            'palm_centroid_pixel': None,
            'normalized_spread': 0.0,
            'annotated_image': rgb_image.copy()
        }

        if len(result.gestures) > 0 and len(result.gestures[0]) > 0:
            # Take the first detected hand's top gesture
            top_gesture = result.gestures[0][0]
            gesture_name = top_gesture.category_name
            confidence = float(top_gesture.score)

            output['gesture_name'] = gesture_name
            output['gesture_confidence'] = confidence
            output['normalized_spread'] = confidence

            # Map Closed_Fist -> pinching
            if gesture_name == 'Closed_Fist':
                output['is_pinching'] = True
            elif gesture_name == 'Open_Palm':
                output['is_pinching'] = False

            # One-hot encoding
            if gesture_name in self.gesture_classes:
                output['gesture_one_hot'][self.gesture_classes.index(gesture_name)] = 1.0

            # Hand landmarks
            if len(result.hand_landmarks) > 0:
                output['hand_landmarks'] = result.hand_landmarks[0]
                output['palm_centroid_pixel'] = self._compute_palm_centroid(result.hand_landmarks[0], rgb_image.shape)
                # Draw centroid marker
                x, y = output['palm_centroid_pixel']
                cv2.circle(output['annotated_image'], (x, y), 10, (0, 0, 255), -1)
                cv2.circle(output['annotated_image'], (x, y), 8, (255, 255, 255), 2)

            # Handedness
            if len(result.handedness) > 0 and len(result.handedness[0]) > 0:
                output['handedness'] = result.handedness[0][0].category_name

            # Draw landmarks
            output['annotated_image'] = self._draw_landmarks(rgb_image, result)

        return output

    def _compute_palm_centroid(self, hand_landmarks, image_shape):
        """Compute pixel coordinates of the palm centroid."""
        palm_indices = [0, 5, 9, 13, 17]  # Wrist + MCPs
        x_norm = np.mean([hand_landmarks[i].x for i in palm_indices])
        y_norm = np.mean([hand_landmarks[i].y for i in palm_indices])
        height, width = image_shape[:2]
        return int(x_norm * width), int(y_norm * height)

    def _draw_landmarks(self, image, result):
        """Draw landmarks and labels on the image."""
        annotated_image = image.copy()
        for idx, landmarks in enumerate(result.hand_landmarks):
            # Convert landmarks to protobuf format
            hand_proto = landmark_pb2.NormalizedLandmarkList()
            hand_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
            ])
            self.drawing_utils.draw_landmarks(
                annotated_image,
                hand_proto,
                self.mp_hands.HAND_CONNECTIONS,
                self.drawing_styles.get_default_hand_landmarks_style(),
                self.drawing_styles.get_default_hand_connections_style()
            )
        return annotated_image

    def get_gesture_classes(self):
        return self.gesture_classes


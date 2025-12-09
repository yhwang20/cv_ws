import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2


# Mariana's code that takes Mediapipe hand landmarks (fingertips etc) and computes if the hand is pinching.

class MediapipeFingerPinch:
    
    def __init__(self, model_path='hand_landmarker.task', 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5,
                 pinch_threshold=0.65):
        
        # Create MediaPipe hand landmarker directly
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.pinch_threshold = pinch_threshold
    
    def __call__(self, rgb_image, depth_image, camera_info):
        # Convert to MediaPipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = self.detector.detect(mp_image)
        
        output = {
            'is_pinching': False,
            'hand_landmarks': result,
            'palm_centroid_pixel': None,
            'normalized_spread': 0.0,
            'annotated_image': rgb_image.copy()
        }
        
        output['annotated_image'] = self._draw_landmarks_on_image(rgb_image, result)
        
        if len(result.hand_landmarks) > 0:
            is_pinching, normalized_spread = self._detect_pinch(result.hand_landmarks[0])
            palm_x, palm_y = self._calculate_palm_centroid(result.hand_landmarks[0], rgb_image.shape)
            
            output['is_pinching'] = is_pinching
            output['normalized_spread'] = normalized_spread
            output['palm_centroid_pixel'] = (palm_x, palm_y)
            
            cv2.circle(output['annotated_image'], (palm_x, palm_y), 10, (0, 0, 255), -1)
            cv2.circle(output['annotated_image'], (palm_x, palm_y), 8, (255, 255, 255), 2)
        
        return output
    
    def _draw_landmarks_on_image(self, rgb_image, detection_result):
        """Draw hand landmarks on the image."""
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

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

            # Add handedness label
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - 10

            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (88, 205, 54),
                1,
                cv2.LINE_AA
            )

        return annotated_image
    
    def _detect_pinch(self, hand_landmarks):
        # fingertip indices: thumb, index, middle, ring, pinky
        fingertip_indices = [4, 8, 12, 16, 20]
        fingertips = [hand_landmarks[i] for i in fingertip_indices]
        
        total_dist = 0.0
        num_pairs = 0
        
        for i in range(len(fingertips)):
            for j in range(i + 1, len(fingertips)):
                tip_i = fingertips[i]
                tip_j = fingertips[j]
                
                dist = np.sqrt(
                    (tip_i.x - tip_j.x)**2 + 
                    (tip_i.y - tip_j.y)**2 +
                    (tip_i.z - tip_j.z)**2
                )
                total_dist += dist
                num_pairs += 1
        
        avg_fingertip_dist = total_dist / num_pairs if num_pairs > 0 else 0.0
        
        # hand size for scale invariance
        wrist = hand_landmarks[0]
        middle_mcp = hand_landmarks[9]
        
        hand_size = np.sqrt(
            (wrist.x - middle_mcp.x)**2 + 
            (wrist.y - middle_mcp.y)**2 +
            (wrist.z - middle_mcp.z)**2
        )
        
        if hand_size > 0:
            normalized_spread = avg_fingertip_dist / hand_size
            is_pinching = normalized_spread < self.pinch_threshold
            return is_pinching, normalized_spread
        
        return False, 0.0
    
    def _calculate_palm_centroid(self, hand_landmarks, image_shape):
        palm_indices = [0, 5, 9, 13, 17]
        
        x_norm = np.mean([hand_landmarks[i].x for i in palm_indices])
        y_norm = np.mean([hand_landmarks[i].y for i in palm_indices])
        
        height, width = image_shape[:2]
        x_pixel = int(x_norm * width)
        y_pixel = int(y_norm * height)
        
        return x_pixel, y_pixel

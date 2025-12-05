import numpy as np
import cv2
from hand_landmarker import HandLandmarkDetector, DetectorConfig, draw_landmarks_on_image


class MediapipeFingerPinch:
    
    def __init__(self, model_path='hand_landmarker.task', 
                 min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5,
                 pinch_threshold=0.65):
        
        config = DetectorConfig(
            model_path=model_path,
            min_tracking_confidence=min_tracking_confidence,
            min_hand_detection_confidence=min_detection_confidence
        )
        
        self.detector = HandLandmarkDetector(config)
        self.pinch_threshold = pinch_threshold
    
    def __call__(self, rgb_image, depth_image, camera_info):
        result = self.detector.detect(rgb_image)
        
        output = {
            'is_pinching': False,
            'hand_landmarks': result,
            'palm_centroid_pixel': None,
            'normalized_spread': 0.0,
            'annotated_image': rgb_image.copy()
        }
        
        output['annotated_image'] = draw_landmarks_on_image(rgb_image, result)
        
        if len(result.hand_landmarks) > 0:
            is_pinching, normalized_spread = self._detect_pinch(result.hand_landmarks[0])
            palm_x, palm_y = self._calculate_palm_centroid(result.hand_landmarks[0], rgb_image.shape)
            
            output['is_pinching'] = is_pinching
            output['normalized_spread'] = normalized_spread
            output['palm_centroid_pixel'] = (palm_x, palm_y)
            
            cv2.circle(output['annotated_image'], (palm_x, palm_y), 10, (0, 0, 255), -1)
            cv2.circle(output['annotated_image'], (palm_x, palm_y), 8, (255, 255, 255), 2)
        
        return output
    
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

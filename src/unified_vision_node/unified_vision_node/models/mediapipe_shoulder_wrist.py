import numpy as np
import cv2
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Hand_and_body_detection'))
from hand_body_detection.core.pose_detector import PoseDetector


class MediapipeShoulderWrist:
    
    def __init__(self, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 model_complexity=2,
                 hand_side='RIGHT',
                 depth_window_size=20):
        
        self.detector = PoseDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        
        self.hand_side = hand_side
        self.wrist_depth_buffer = deque(maxlen=depth_window_size)
        self.shoulder_depth_buffer = deque(maxlen=depth_window_size)
    
    def __call__(self, rgb_image, depth_image, camera_info):
        results = self.detector.detect(rgb_image)
        
        output = {
            'wrist_position': None,
            'shoulder_position': None,
            'wrist_pixel': None,
            'shoulder_pixel': None,
            'pose_landmarks': results,
            'annotated_image': rgb_image.copy()
        }
        
        self.detector.draw_landmarks(output['annotated_image'], results)
        
        positions = self.detector.get_wrist_shoulder_positions(results, self.hand_side)
        
        if positions is not None:
            wrist_norm = positions['wrist']
            shoulder_norm = positions['shoulder']
            
            height, width = rgb_image.shape[:2]
            wrist_pixel = (int(wrist_norm[0] * width), int(wrist_norm[1] * height))
            shoulder_pixel = (int(shoulder_norm[0] * width), int(shoulder_norm[1] * height))
            
            output['wrist_pixel'] = wrist_pixel
            output['shoulder_pixel'] = shoulder_pixel
            
            # draw markers
            cv2.circle(output['annotated_image'], wrist_pixel, 12, (255, 0, 0), -1)
            cv2.circle(output['annotated_image'], wrist_pixel, 10, (255, 255, 255), 2)
            cv2.putText(output['annotated_image'], 'WRIST', (wrist_pixel[0] + 15, wrist_pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.circle(output['annotated_image'], shoulder_pixel, 12, (0, 255, 0), -1)
            cv2.circle(output['annotated_image'], shoulder_pixel, 10, (255, 255, 255), 2)
            cv2.putText(output['annotated_image'], 'SHOULDER', (shoulder_pixel[0] + 15, shoulder_pixel[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # calculate 3d positions with depth
            if depth_image is not None and camera_info is not None:
                wrist_depth_raw = self._get_depth_at_pixel(depth_image, wrist_pixel[0], wrist_pixel[1])
                shoulder_depth_raw = self._get_depth_at_pixel(depth_image, shoulder_pixel[0], shoulder_pixel[1])
                
                if wrist_depth_raw is not None:
                    self.wrist_depth_buffer.append(wrist_depth_raw)
                
                if shoulder_depth_raw is not None:
                    self.shoulder_depth_buffer.append(shoulder_depth_raw)
                
                wrist_depth_avg = self._get_averaged_depth(self.wrist_depth_buffer)
                shoulder_depth_avg = self._get_averaged_depth(self.shoulder_depth_buffer)
                
                if wrist_depth_avg is not None:
                    output['wrist_position'] = self._pixel_to_3d(
                        wrist_pixel[0], wrist_pixel[1], wrist_depth_avg, camera_info
                    )
                
                if shoulder_depth_avg is not None:
                    output['shoulder_position'] = self._pixel_to_3d(
                        shoulder_pixel[0], shoulder_pixel[1], shoulder_depth_avg, camera_info
                    )
        
        return output
    
    def _get_depth_at_pixel(self, depth_image, pixel_x, pixel_y):
        if depth_image is None:
            return None
        
        depth_height, depth_width = depth_image.shape[:2]
        
        if pixel_x < 0 or pixel_x >= depth_width or pixel_y < 0 or pixel_y >= depth_height:
            return None
        
        depth_value = depth_image[pixel_y, pixel_x]
        
        if depth_image.dtype == np.uint16:
            depth_meters = float(depth_value) / 1000.0
        else:
            depth_meters = float(depth_value)
        
        if depth_meters <= 0 or np.isnan(depth_meters):
            return None
        
        return depth_meters
    
    def _get_averaged_depth(self, depth_buffer):
        if len(depth_buffer) == 0:
            return None
        
        return np.mean(depth_buffer)
    
    def _pixel_to_3d(self, pixel_x, pixel_y, depth_value, camera_info):
        if camera_info is None:
            return None
        
        if depth_value <= 0 or np.isnan(depth_value):
            return None
        
        x = (pixel_x - camera_info['cx']) * depth_value / camera_info['fx']
        y = (pixel_y - camera_info['cy']) * depth_value / camera_info['fy']
        z = depth_value
        
        return (x, y, z)

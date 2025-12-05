import mediapipe as mp
import numpy as np
import cv2


class PoseDetector:
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
    
    def detect(self, image):
        results = self.pose.process(image)
        return results
    
    def get_landmark_position(self, results, landmark_index):
        if not results.pose_landmarks:
            return None
        
        landmark = results.pose_landmarks.landmark[landmark_index]
        return (landmark.x, landmark.y, landmark.z)
    
    def get_wrist_shoulder_positions(self, results, hand='RIGHT'):
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        
        if hand.upper() == 'RIGHT':
            wrist_idx = self.mp_pose.PoseLandmark.RIGHT_WRIST.value
            shoulder_idx = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            elbow_idx = self.mp_pose.PoseLandmark.RIGHT_ELBOW.value
        else:
            wrist_idx = self.mp_pose.PoseLandmark.LEFT_WRIST.value
            shoulder_idx = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
            elbow_idx = self.mp_pose.PoseLandmark.LEFT_ELBOW.value
        
        wrist = landmarks[wrist_idx]
        shoulder = landmarks[shoulder_idx]
        elbow = landmarks[elbow_idx]
        
        return {
            'wrist': (wrist.x, wrist.y, wrist.z),
            'shoulder': (shoulder.x, shoulder.y, shoulder.z),
            'elbow': (elbow.x, elbow.y, elbow.z)
        }
    
    def draw_landmarks(self, image, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return image
    
    def close(self):
        self.pose.close()


def calculate_angle(a, b, c):
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
              np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

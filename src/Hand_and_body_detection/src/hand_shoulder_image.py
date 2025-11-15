import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
#File to test hand and shoulder detection in a static image

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles

def calculate_angle(a, b, c):
    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
          np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
#Grab images from data folder
script_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE_FILES = [
    os.path.join(script_dir, "..", "data", "demo.jpg"),
    os.path.join(script_dir, "..", "data", "testing2.jpg")
]

# Create output folder if it doesn't exist
OUTPUT_DIR = os.path.join(script_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BG_COLOR = (192, 192, 192) # gray
annotated_image = []
counter = 0
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.5) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    if image is None:
        print(f"ERROR: Cannot read image: {file}")
        continue
    else:
        print(f"Loaded image: {file} with shape {image.shape}")
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
        print(f"No pose detected in: {file}")
        continue
    else:
        print(f"Pose detected! {len(results.pose_landmarks.landmark)} landmarks found.")
    annotated_image.append(image.copy())

    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image[counter] = np.where(condition, annotated_image[counter], bg_image)
    # Grab coordinates of landmarks and angles
    landmarks = results.pose_landmarks.landmark
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z]
    base_angle = calculate_angle(right_hip, shoulder, elbow)
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    cv2.putText(annotated_image[counter], str(int(base_angle)),
            tuple(np.multiply(shoulder[:2], [image.shape[1], image.shape[0]]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                )
    cv2.putText(annotated_image[counter], str(int(elbow_angle)),
            tuple(np.multiply(elbow[:2], [image.shape[1], image.shape[0]]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                )
    mp_drawing.draw_landmarks(
        annotated_image[counter],
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    out_path = os.path.join(OUTPUT_DIR, f"annotated_{idx}.png")
    cv2.imwrite(out_path, annotated_image[counter])
    print(f"Saved annotated image: {out_path}")
    counter +=1
    print(f"Shoulder Coordinate of image{counter}:", shoulder)
    print(f"Elbow Coordinate of image{counter}:", elbow)
    print(f"Wrist Coordinate of image{counter}:", wrist)
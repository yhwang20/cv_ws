from ultralytics import YOLO
import os
import cv2
import numpy as np

model = YOLO('yolo11n-pose.pt') 
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame, stream=True)
    for r in results:
        annotated_frame = r.plot()
        print("hello")
        keypoints = r.keypoints.xy
        right_shoulder = keypoints[0][6]
        right_wrist = keypoints[0][10]
    
    cv2.circle(annotated_frame, (int(right_shoulder[0]), int(right_shoulder[1])), 5, (0, 255, 0), -1)
    cv2.circle(annotated_frame, (int(right_wrist[0]), int(right_wrist[1])), 5, (0, 0, 255), -1)
    cv2.putText(annotated_frame, 'Right Shoulder', (int(right_shoulder[0]) + 10, int(right_shoulder[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(annotated_frame, 'Right Wrist', (int(right_wrist[0]) + 10, int(right_wrist[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("YOLO Pose Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
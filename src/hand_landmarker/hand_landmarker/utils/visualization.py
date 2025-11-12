"""Visualization utilities for hand landmarks."""

import numpy as np
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
from typing import Optional, Tuple


# Default visualization constants
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(
    rgb_image: np.ndarray,
    detection_result: vision.HandLandmarkerResult,
    margin: int = MARGIN,
    font_size: float = FONT_SIZE,
    font_thickness: int = FONT_THICKNESS,
    text_color: Tuple[int, int, int] = HANDEDNESS_TEXT_COLOR,
) -> np.ndarray:
    """Draw hand landmarks and handedness on an image.
    
    Args:
        rgb_image: Input image in RGB format (numpy array)
        detection_result: HandLandmarkerResult from MediaPipe detection
        margin: Margin in pixels for text placement
        font_size: Font size for handedness text
        font_thickness: Font thickness for handedness text
        text_color: RGB color tuple for handedness text
    
    Returns:
        Annotated image with landmarks and handedness labels drawn
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x, y=landmark.y, z=landmark.z
            ) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        # Get the top left corner of the detected hand's bounding box
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - margin

        # Draw handedness (left or right hand) on the image
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            font_size,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )

    return annotated_image


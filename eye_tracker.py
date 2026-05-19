import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join("models", "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Missing model file at 'models/face_landmarker.task'. Download it first."
    )

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip for selfie-view, then convert to RGB for MediaPipe.
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = face_landmarker.detect(mp_image)

    if result.face_landmarks:
        for face_landmarks in result.face_landmarks:
            # MediaPipe's specific landmark indices for the left and right irises
            left_iris_indices = [474, 475, 476, 477]
            right_iris_indices = [469, 470, 471, 472]
            
            h, w, _ = frame.shape
            
            # Draw green dots on the Left Iris
            for index in left_iris_indices:
                point = face_landmarks[index]
                x, y = int(point.x * w), int(point.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                
            # Draw red dots on the Right Iris
            for index in right_iris_indices:
                point = face_landmarks[index]
                x, y = int(point.x * w), int(point.y * h)
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # Show the video feed
    cv2.imshow('Drishti - Eye Tracking PoC', frame)

    # Press 'q' to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()
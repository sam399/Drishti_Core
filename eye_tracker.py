import os

import cv2
import mediapipe as mp
import numpy as np
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

# Calibration settings
CALIBRATION_POINTS = [
    (0.1, 0.1),
    (0.9, 0.1),
    (0.5, 0.5),
    (0.1, 0.9),
    (0.9, 0.9),
]
HOLD_FRAMES = 20
SAMPLES_PER_POINT = 15


def get_iris_center(face_landmarks):
    left_iris_indices = [474, 475, 476, 477]
    right_iris_indices = [469, 470, 471, 472]
    indices = left_iris_indices + right_iris_indices
    xs = [face_landmarks[i].x for i in indices]
    ys = [face_landmarks[i].y for i in indices]
    return float(np.mean(xs)), float(np.mean(ys))


def fit_affine(samples):
    a = np.array([[x, y, 1.0] for x, y, _, _ in samples], dtype=np.float32)
    bx = np.array([tx for _, _, tx, _ in samples], dtype=np.float32)
    by = np.array([ty for _, _, _, ty in samples], dtype=np.float32)
    coef_x, _, _, _ = np.linalg.lstsq(a, bx, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(a, by, rcond=None)
    return coef_x, coef_y


def apply_affine(coef_x, coef_y, x, y):
    sx = float(coef_x[0] * x + coef_x[1] * y + coef_x[2])
    sy = float(coef_y[0] * x + coef_y[1] * y + coef_y[2])
    return max(0.0, min(1.0, sx)), max(0.0, min(1.0, sy))


def reset_calibration():
    return 0, 0, [], [], False, None, None


# Open the webcam
cap = cv2.VideoCapture(0)

calibration_index, hold_count, current_samples, all_samples, calibrated, affine_x, affine_y = (
    reset_calibration()
)

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

    h, w, _ = frame.shape

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]
        iris_x, iris_y = get_iris_center(face_landmarks)

        if not calibrated:
            target_x, target_y = CALIBRATION_POINTS[calibration_index]
            target_px = int(target_x * w)
            target_py = int(target_y * h)
            cv2.circle(frame, (target_px, target_py), 12, (255, 255, 0), -1)
            progress = min(1.0, hold_count / float(HOLD_FRAMES))
            cv2.ellipse(
                frame,
                (target_px, target_py),
                (20, 20),
                -90,
                0,
                int(360 * progress),
                (0, 255, 255),
                3,
            )
            cv2.putText(
                frame,
                f"Calibrating {calibration_index + 1}/{len(CALIBRATION_POINTS)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if hold_count < HOLD_FRAMES:
                hold_count += 1
            else:
                current_samples.append((iris_x, iris_y))
                if len(current_samples) >= SAMPLES_PER_POINT:
                    mean_x = float(np.mean([s[0] for s in current_samples]))
                    mean_y = float(np.mean([s[1] for s in current_samples]))
                    all_samples.append((mean_x, mean_y, target_x, target_y))
                    current_samples = []
                    hold_count = 0
                    calibration_index += 1

                    if calibration_index >= len(CALIBRATION_POINTS):
                        affine_x, affine_y = fit_affine(all_samples)
                        calibrated = True
        else:
            gaze_x, gaze_y = apply_affine(affine_x, affine_y, iris_x, iris_y)
            gaze_px = int(gaze_x * w)
            gaze_py = int(gaze_y * h)
            cv2.circle(frame, (gaze_px, gaze_py), 10, (0, 255, 255), -1)
    else:
        if not calibrated:
            hold_count = 0
            current_samples = []
        cv2.putText(
            frame,
            "No face detected",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    cv2.putText(
        frame,
        "Press r to recalibrate, q to quit",
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )

    # Show the video feed
    cv2.imshow('Drishti - Eye Tracking PoC', frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('r'):
        calibration_index, hold_count, current_samples, all_samples, calibrated, affine_x, affine_y = (
            reset_calibration()
        )
    # Press 'q' to exit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()
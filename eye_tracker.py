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


def get_relative_gaze_vector(face_landmarks, corner_history=None, alpha_corner=0.08, prev_gaze=None, alpha_gaze=0.35):
    if len(face_landmarks) < 478:
        # Fallback to standard tracking if refined iris landmarks are missing
        return (0.0, 0.0), prev_gaze

    # Smoothed corner coordinates
    smoothed = {}
    corner_indices = [33, 133, 263, 362]
    
    for idx in corner_indices:
        raw_pt = face_landmarks[idx]
        if corner_history is not None and idx in corner_history:
            prev_x, prev_y = corner_history[idx]
            smooth_x = alpha_corner * raw_pt.x + (1.0 - alpha_corner) * prev_x
            smooth_y = alpha_corner * raw_pt.y + (1.0 - alpha_corner) * prev_y
        else:
            smooth_x = raw_pt.x
            smooth_y = raw_pt.y
        if corner_history is not None:
            corner_history[idx] = (smooth_x, smooth_y)
        smoothed[idx] = (smooth_x, smooth_y)

    # Left eye landmarks: outer (33), inner (133), iris (474, 475, 476, 477)
    l_outer_x, l_outer_y = smoothed[33]
    l_inner_x, l_inner_y = smoothed[133]
    l_iris_indices = [474, 475, 476, 477]
    
    l_center_x = (l_outer_x + l_inner_x) / 2.0
    l_center_y = (l_outer_y + l_inner_y) / 2.0
    l_width = ((l_outer_x - l_inner_x) ** 2 + (l_outer_y - l_inner_y) ** 2) ** 0.5
    
    l_iris_x = float(np.mean([face_landmarks[i].x for i in l_iris_indices]))
    l_iris_y = float(np.mean([face_landmarks[i].y for i in l_iris_indices]))
    
    l_off_x = (l_iris_x - l_center_x) / l_width if l_width > 0.0 else 0.0
    l_off_y = (l_iris_y - l_center_y) / l_width if l_width > 0.0 else 0.0

    # Right eye landmarks: outer (263), inner (362), iris (469, 470, 471, 472)
    r_outer_x, r_outer_y = smoothed[263]
    r_inner_x, r_inner_y = smoothed[362]
    r_iris_indices = [469, 470, 471, 472]
    
    r_center_x = (r_outer_x + r_inner_x) / 2.0
    r_center_y = (r_outer_y + r_inner_y) / 2.0
    r_width = ((r_outer_x - r_inner_x) ** 2 + (r_outer_y - r_inner_y) ** 2) ** 0.5
    
    r_iris_x = float(np.mean([face_landmarks[i].x for i in r_iris_indices]))
    r_iris_y = float(np.mean([face_landmarks[i].y for i in r_iris_indices]))
    
    r_off_x = (r_iris_x - r_center_x) / r_width if r_width > 0.0 else 0.0
    r_off_y = (r_iris_y - r_center_y) / r_width if r_width > 0.0 else 0.0

    # Average left and right relative gaze vectors
    raw_gaze_x = (l_off_x + r_off_x) / 2.0
    raw_gaze_y = (l_off_y + r_off_y) / 2.0

    # Gaze vector low pass smoothing
    if prev_gaze is not None and prev_gaze[0] is not None and prev_gaze[1] is not None:
        smooth_gaze_x = alpha_gaze * raw_gaze_x + (1.0 - alpha_gaze) * prev_gaze[0]
        smooth_gaze_y = alpha_gaze * raw_gaze_y + (1.0 - alpha_gaze) * prev_gaze[1]
    else:
        smooth_gaze_x = raw_gaze_x
        smooth_gaze_y = raw_gaze_y

    new_gaze = (float(smooth_gaze_x), float(smooth_gaze_y))
    return new_gaze, new_gaze


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
corner_history = {}
prev_gaze = [None, None]

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
        (iris_x, iris_y), _ = get_relative_gaze_vector(
            face_landmarks,
            corner_history=corner_history,
            alpha_corner=0.08,
            prev_gaze=prev_gaze,
            alpha_gaze=0.35
        )
        prev_gaze[0] = iris_x
        prev_gaze[1] = iris_y

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
        corner_history.clear()
        prev_gaze = [None, None]
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
        corner_history.clear()
        prev_gaze = [None, None]
    # Press 'q' to exit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_landmarker.close()
import argparse
import asyncio
import copy
import json
import os
import queue
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import websockets
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join("models", "face_landmarker.task")

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
TARGET_SEND_HZ = 30
INFER_WIDTH = 320


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


def reset_calibration_state():
    return {
        "calibration_index": 0,
        "hold_count": 0,
        "current_samples": [],
        "all_samples": [],
        "calibrated": False,
        "affine_x": None,
        "affine_y": None,
        "gaze_x": None,
        "gaze_y": None,
        "face_detected": False,
    }


def create_face_landmarker():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Missing model file at 'models/face_landmarker.task'. Download it first."
        )

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _inference_worker(
    frame_queue: queue.Queue,
    shared_state: dict,
    state_lock: threading.Lock,
    stop_event: threading.Event,
    reset_event: threading.Event,
) -> None:
    face_landmarker = create_face_landmarker()
    state = reset_calibration_state()
    corner_history = {}
    prev_gaze = [None, None]

    while not stop_event.is_set():
        try:
            rgb_frame = frame_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if reset_event.is_set():
            state = reset_calibration_state()
            corner_history.clear()
            prev_gaze = [None, None]
            reset_event.clear()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = face_landmarker.detect(mp_image)
        face_detected = bool(result.face_landmarks)

        state["face_detected"] = face_detected

        if face_detected:
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

            if not state["calibrated"]:
                target_x, target_y = CALIBRATION_POINTS[state["calibration_index"]]
                if state["hold_count"] < HOLD_FRAMES:
                    state["hold_count"] += 1
                else:
                    state["current_samples"].append((iris_x, iris_y))
                    if len(state["current_samples"]) >= SAMPLES_PER_POINT:
                        mean_x = float(
                            np.mean([s[0] for s in state["current_samples"]])
                        )
                        mean_y = float(
                            np.mean([s[1] for s in state["current_samples"]])
                        )
                        state["all_samples"].append((mean_x, mean_y, target_x, target_y))
                        state["current_samples"] = []
                        state["hold_count"] = 0
                        state["calibration_index"] += 1

                        if state["calibration_index"] >= len(CALIBRATION_POINTS):
                            state["affine_x"], state["affine_y"] = fit_affine(
                                state["all_samples"]
                            )
                            state["calibrated"] = True
            else:
                gaze_x, gaze_y = apply_affine(
                    state["affine_x"], state["affine_y"], iris_x, iris_y
                )
                state["gaze_x"] = gaze_x
                state["gaze_y"] = gaze_y
        else:
            state["gaze_x"] = None
            state["gaze_y"] = None
            corner_history.clear()
            prev_gaze = [None, None]
            if not state["calibrated"]:
                state["hold_count"] = 0
                state["current_samples"] = []

        with state_lock:
            shared_state.update(state)

    face_landmarker.close()


def _ws_sender_worker(
    ws_url: str,
    ws_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
    async def send_loop():
        loop = asyncio.get_running_loop()
        while not stop_event.is_set():
            try:
                async with websockets.connect(
                    ws_url, ping_interval=20, ping_timeout=20
                ) as websocket:
                    print("Streamer connected to backend", flush=True)
                    while not stop_event.is_set():
                          try:
                              # Offload blocking queue read to thread pool (0% CPU idle, zero GIL contention)
                              payload = await loop.run_in_executor(None, ws_queue.get)
                          except Exception:
                              continue

                          if stop_event.is_set():
                              break

                          try:
                              await websocket.send(json.dumps(payload))
                          except Exception as exc:
                              print(f"WebSocket send failed: {exc}", flush=True)
                              break
            except Exception as exc:
                print(f"WebSocket connection error: {exc}", flush=True)
                # Wait 1.0s safely checking stop_event
                for _ in range(10):
                    if stop_event.is_set():
                        break
                    await asyncio.sleep(0.1)

    asyncio.run(send_loop())


def run(ws_url: str) -> None:
    cap = cv2.VideoCapture(0)
    frame_queue: queue.Queue = queue.Queue(maxsize=1)
    ws_queue: queue.Queue = queue.Queue(maxsize=2)
    state_lock = threading.Lock()
    shared_state = reset_calibration_state()
    stop_event = threading.Event()
    reset_event = threading.Event()

    # 1. Start inference thread
    worker = threading.Thread(
        target=_inference_worker,
        args=(frame_queue, shared_state, state_lock, stop_event, reset_event),
        daemon=True,
    )
    worker.start()

    # 2. Start WebSocket sender thread
    sender = threading.Thread(
        target=_ws_sender_worker,
        args=(ws_url, ws_queue, stop_event),
        daemon=True,
    )
    sender.start()

    target_interval = 1.0 / TARGET_SEND_HZ
    last_send = 0.0
    last_fps_time = time.monotonic()
    last_log_time = time.monotonic()
    frame_count = 0
    send_count = 0
    fps = 0.0

    try:
        while cap.isOpened() and not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.", flush=True)
                continue

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            scale = INFER_WIDTH / float(w)
            infer_h = int(h * scale)
            resized = cv2.resize(frame, (INFER_WIDTH, infer_h))
            rgb_small = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put_nowait(rgb_small)

            with state_lock:
                state = copy.deepcopy(shared_state)

            if state["face_detected"]:
                if state["calibrated"] and state["gaze_x"] is not None:
                    gaze_px = int(state["gaze_x"] * w)
                    gaze_py = int(state["gaze_y"] * h)
                    cv2.circle(
                        frame, (gaze_px, gaze_py), 10, (0, 255, 255), -1
                    )
                else:
                    safe_index = min(
                        state["calibration_index"], len(CALIBRATION_POINTS) - 1
                    )
                    target_x, target_y = CALIBRATION_POINTS[safe_index]
                    target_px = int(target_x * w)
                    target_py = int(target_y * h)
                    cv2.circle(
                        frame, (target_px, target_py), 12, (255, 255, 0), -1
                    )
                    progress = min(1.0, state["hold_count"] / float(HOLD_FRAMES))
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
                        f"Calibrating {state['calibration_index'] + 1}/{len(CALIBRATION_POINTS)}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            frame_count += 1
            now = time.monotonic()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            if now - last_send >= target_interval:
                if state["calibrated"] and state["gaze_x"] is not None:
                    payload = {
                        "type": "gaze",
                        "x": state["gaze_x"],
                        "y": state["gaze_y"],
                    }
                else:
                    safe_index = min(
                        state["calibration_index"],
                        len(CALIBRATION_POINTS) - 1,
                    )
                    payload = {
                        "type": "status",
                        "calibrated": state["calibrated"],
                        "face_detected": state["face_detected"],
                        "step": safe_index + 1,
                        "total": len(CALIBRATION_POINTS),
                    }
                
                # Push payload into ws_queue (discard old item if full to avoid latency)
                if ws_queue.full():
                    try:
                        ws_queue.get_nowait()
                    except queue.Empty:
                        pass
                ws_queue.put_nowait(payload)
                send_count += 1
                last_send = now

            if now - last_log_time >= 2.0:
                print(
                    "Streamer status: "
                    f"face={state['face_detected']} "
                    f"calibrated={state['calibrated']} "
                    f"sent={send_count}",
                    flush=True
                )
                last_log_time = now

            cv2.putText(
                frame,
                "Press r to recalibrate, q to quit",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )
            cv2.putText(
                frame,
                f"Face: {'yes' if state['face_detected'] else 'no'} | FPS: {fps:.1f}",
                (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )

            cv2.imshow("Drishti - Eye Tracking Stream", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("r"):
                reset_event.set()
            if key == ord("q"):
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream gaze to a WebSocket backend.")
    parser.add_argument(
        "--ws",
        default="ws://127.0.0.1:8000/ws/producer",
        help="WebSocket URL for the gaze producer endpoint.",
    )
    args = parser.parse_args()

    if not args.ws.startswith("ws://") and not args.ws.startswith("wss://"):
        raise ValueError("WebSocket URL must start with ws:// or wss://")

    run(args.ws)


if __name__ == "__main__":
    main()

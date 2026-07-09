import cv2
import numpy as np
import time
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Settings ──────────────────────────────────────────────────────────────────
DURATION    = 10
BRUSH_COLOR = (0, 100, 255)   # BGR orange
BRUSH_THICK = 6

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

INDEX_TIP = 8
INDEX_PIP = 6

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

# ── State ──────────────────────────────────────────────────────────────────────
canvas         = None
prev_pt        = None
running        = False
start_time     = None
_latest_result = None


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmark model (~25 MB) — one time only...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.\n")


def on_result(result, output_image, timestamp_ms):
    global _latest_result
    _latest_result = result


def get_tip(frame_w, frame_h):
    """Return ((x,y), finger_up) or (None, False)."""
    if not _latest_result or not _latest_result.hand_landmarks:
        return None, False
    lms       = _latest_result.hand_landmarks[0]
    tip       = lms[INDEX_TIP]
    pip       = lms[INDEX_PIP]
    x, y      = int(tip.x * frame_w), int(tip.y * frame_h)
    finger_up = tip.y < pip.y
    return (x, y), finger_up


def draw_skeleton(frame, frame_w, frame_h):
    if not _latest_result or not _latest_result.hand_landmarks:
        return
    for lms in _latest_result.hand_landmarks:
        pts = [(int(l.x * frame_w), int(l.y * frame_h)) for l in lms]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (80, 80, 80), 1)
        for p in pts:
            cv2.circle(frame, p, 3, (160, 160, 160), -1)


def draw_ui(frame, time_left):
    h, w = frame.shape[:2]
    if running:
        ratio = max(0.0, time_left / DURATION)
        bar_w = int(w * ratio)
        color = (60, 200, 60) if ratio > 0.5 else (0, 165, 255) if ratio > 0.25 else (0, 0, 220)
        cv2.rectangle(frame, (0, h - 12), (bar_w, h), color, -1)
        cv2.putText(frame, f"{time_left:.1f}s", (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, "curl finger = lift pen", (w - 205, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "S = start  |  C = clear  |  Q = quit",
                    (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)


def main():
    global canvas, prev_pt, running, start_time

    download_model()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Could not read from webcam.")
        cap.release()
        return

    canvas = np.zeros_like(frame)

    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options   = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        result_callback=on_result,
    )

    print("Virtual Pen ready.")
    print("  Extend index finger = draw")
    print("  Curl finger down    = lift pen")
    print("  S = start 10-sec session  |  C = clear  |  Q = quit\n")

    ts = 0
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            if canvas.shape != frame.shape:
                canvas = np.zeros_like(frame)

            ts += 1
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarker.detect_async(mp_img, ts)

            time_left = 0
            if running:
                time_left = DURATION - (time.time() - start_time)
                if time_left <= 0:
                    running = False
                    prev_pt = None
                    time_left = 0
                    print("Time's up!  S = draw again  |  C = clear")

            pt, finger_up = get_tip(w, h)

            if pt and running and finger_up:
                if prev_pt:
                    cv2.line(canvas, prev_pt, pt, BRUSH_COLOR, BRUSH_THICK)
                else:
                    cv2.circle(canvas, pt, BRUSH_THICK // 2, BRUSH_COLOR, -1)
                prev_pt = pt
            else:
                prev_pt = None

            draw_skeleton(frame, w, h)
            output = cv2.addWeighted(frame, 1.0, canvas, 1.0, 0)

            if pt:
                cv2.circle(output, pt, 10, (255, 255, 255), 1)
                cv2.circle(output, pt,  3, BRUSH_COLOR,     -1)

            draw_ui(output, time_left)
            cv2.imshow("Virtual Pen", output)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not running:
                running    = True
                start_time = time.time()
                prev_pt    = None
                print("Go! Extend your index finger to draw.")
            elif key == ord('c'):
                canvas  = np.zeros_like(frame)
                running = False
                prev_pt = None
                print("Canvas cleared.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Virtual Pen with MediaPipe Hand Tracking

## Overview
A Python-based real-time virtual pen that uses your **webcam and index finger** to draw on screen — no colored markers needed. Powered by MediaPipe's hand landmark model, it detects and tracks your fingertip with 21-point hand skeleton detection and lets you draw freely in a timed session.

---

## Architecture
* **OpenCV** → webcam capture, canvas drawing, frame compositing, window display
* **MediaPipe** → real-time hand landmark detection (21 points per hand)
* **NumPy** → blank canvas creation and frame blending
* **urllib** → one-time auto-download of the MediaPipe hand model

---

## How It Works
* Webcam feed is captured and mirrored horizontally each frame
* Each frame is sent asynchronously to MediaPipe's `HandLandmarker` in `LIVE_STREAM` mode
* MediaPipe detects 21 landmarks on your hand and returns results via a background callback
* **Landmark #8** (index fingertip) is tracked for position
* **Landmark #6** (index PIP joint) is compared to #8 to detect finger state
* If tip is **above** the PIP joint → finger extended → pen draws
* If tip is **below** the PIP joint → finger curled → pen lifts (new stroke on next extension)
* Drawn strokes are painted onto a separate black canvas and blended over the live feed

---

## Finger State Logic

| State | Condition | Pen Behavior |
|-------|-----------|--------------|
| Extended | `tip.y < pip.y` | Draws a line to current position |
| Curled | `tip.y >= pip.y` | Lifts pen, resets stroke |
| Not detected | No hand in frame | Pen lifted |

---

## Key Design Decisions

### Async detection
MediaPipe runs in `LIVE_STREAM` mode with a result callback — detection happens in the background without blocking the camera loop, keeping the feed smooth.

### Separate canvas layer
Drawing happens on a dedicated `numpy` black image, not directly on the camera frame. This means the live feed and the drawing are independent — the drawing persists across frames while the camera updates every frame.

### Stroke reset on curl
When `prev_pt` is set to `None` on finger curl, the next extension starts a brand new stroke with no connecting line — mimicking lifting a real pen off paper.

---

## Files
* `virtual_pen.py` → main script — webcam loop, hand detection, drawing logic, timer, UI overlay
* `hand_landmarker.task` → MediaPipe model file (auto-downloaded on first run, ~25 MB)

---

## Requirements

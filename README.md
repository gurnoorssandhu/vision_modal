# vision_modal

Physics-informed, low-latency computer vision for obstacle detection, motion
**prediction**, and avoidance — built to run on a laptop webcam first, then a
Raspberry Pi 4 (Arducam OV5647) as physical AI.

## What it does

A decoupled, multi-threaded pipeline:

```
camera (threaded, newest-frame)
   -> MediaPipe object detection (EfficientDet-Lite, tflite/XNNPACK)
   -> multi-object tracking (IoU + persistent IDs)
   -> Kalman filter (smooth state: position + velocity in [x, y, looming])
   -> RK4 trajectory prediction (integrate the motion ODE forward)
   -> collision risk (looming time-to-collision + ego danger corridor)
   -> avoidance command (STOP / LEFT / RIGHT / CLEAR + steer vector)
   -> overlay / annotated stream
```

Plus an **async OpenAI scene-reasoning thread** (vision) that adds high-level
hazard understanding ~1 Hz **without ever blocking the reactive loop**.

Design notes:
- **Physics-informed:** each object is a Newtonian state; a Kalman filter estimates
  it, and classical **RK4** integrates the motion model forward to predict future
  positions. Swap `constant_velocity` for `drag` (nonlinear) in `config.py` and RK4
  handles it unchanged.
- **Monocular for now:** depth is a relative looming cue, not metric. The
  `perception/depth.py` `DepthChannel` interface is the seam for a stereo / depth
  camera (OAK-D, RealSense) later — physics/planning won't change.
- **Scene model stays off the hot path:** the reactive loop runs at full FPS even
  with no network; the scene line just goes stale.

## Setup (laptop)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# detector model (~4.5 MB)
curl -L -o models/efficientdet_lite0.tflite \
  https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite

export OPENAI_API_KEY=sk-...   # optional; enables scene reasoning
python app.py
```

Walk toward the webcam: boxes track you, magenta ghost boxes project where RK4
predicts you'll be, the corridor turns red as TTC drops, and the HUD prints a
`STOP` / `LEFT` / `RIGHT` command. Press `q` to quit. Run `python app.py --no-llm`
to skip the scene-reasoning thread.

## Raspberry Pi 4

```bash
sudo apt install -y python3-picamera2
# (use a venv with --system-site-packages so picamera2 is importable, or install
#  opencv/mediapipe/openai into the system python)
python app.py --source picamera --headless
# view annotated stream at http://<pi-ip>:8000   (e.g. http://100.125.23.18:8000)
```

Tune for the Pi in `config.py`: lower `width/height`, raise `detect_every`
(detect every Nth frame, Kalman coasts between).

## Layout

| path | role |
|---|---|
| `camera/` | `CameraSource` backends (webcam, picamera; stereo later) |
| `perception/` | detector, tracker, depth channel |
| `physics/` | state, Kalman, motion models, RK4, collision |
| `planning/` | risk -> avoidance command (+ motor seam) |
| `reasoning/` | async OpenAI scene reasoning |
| `viz/` | overlay drawing + MJPEG server |
| `app.py` | wires the threads together |
| `stream.py` | standalone raw MJPEG camera server (Pi sanity check) |

## Roadmap

- M8: stereo/depth-cam source behind `DepthChannel`; motor actuation behind
  `planning/avoidance.py:to_motor`.

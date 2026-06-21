"""vision_modal — physics-informed real-time obstacle detection + prediction.

Wires the decoupled pipeline:
  camera (threaded)  ->  detect -> track -> Kalman -> RK4 predict
                         -> collision risk -> avoidance command -> overlay
  + an async OpenAI scene-reasoning thread that never blocks the hot loop.

Run on a laptop webcam:   python app.py
Run headless on the Pi:   python app.py --source picamera --headless
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

import config as cfg_mod
from perception.async_detector import AsyncDetector
from perception.tracker import Tracker
from perception.depth import MonoLoomingDepth
from physics import rk4
from physics.motion_models import get_model
from physics import collision
from planning import avoidance
from reasoning.scene_llm import SceneReasoner
from viz import overlay


def build_camera(cfg):
    if cfg.source == "picamera":
        from camera.picamera import PiCameraSource
        return PiCameraSource(cfg.width, cfg.height, cfg.flip)
    from camera.webcam import WebcamSource
    return WebcamSource(cfg.cam_index, cfg.width, cfg.height, cfg.flip)


def parse_args(cfg):
    p = argparse.ArgumentParser(description="vision_modal pipeline")
    p.add_argument("--source", choices=["webcam", "picamera"], default=cfg.source)
    p.add_argument("--headless", action="store_true", help="serve MJPEG instead of a window")
    p.add_argument("--no-llm", action="store_true", help="disable OpenAI scene reasoning")
    p.add_argument("--backend", choices=["litert", "mediapipe"], default=cfg.detector_backend)
    p.add_argument("--model", default=None, help="override detector model path")
    p.add_argument("--threads", type=int, default=cfg.num_threads, help="LiteRT detector threads")
    p.add_argument("--servo", action="store_true", help="enable haptic servo (Pi GPIO)")
    p.add_argument("--servo-pin", type=int, default=cfg.servo_pin, help="BCM pin for servo signal")
    p.add_argument("--profile", action="store_true", help="print per-stage timing")
    p.add_argument("--width", type=int, default=cfg.width)
    p.add_argument("--height", type=int, default=cfg.height)
    p.add_argument("--flip", action="store_true", default=cfg.flip)
    a = p.parse_args()
    cfg.source = a.source
    cfg.headless = a.headless
    cfg.detector_backend = a.backend
    if a.model:
        cfg.model_path = a.model
        cfg.mediapipe_model_path = a.model
    cfg.profile = a.profile
    cfg.num_threads = a.threads
    cfg.servo_enabled = a.servo
    cfg.servo_pin = a.servo_pin
    cfg.width, cfg.height, cfg.flip = a.width, a.height, a.flip
    if a.no_llm:
        cfg.llm_enabled = False
    return cfg


def main() -> int:
    cfg = parse_args(cfg_mod.load())

    if cfg.detector_backend == "mediapipe":
        model_path = cfg.mediapipe_model_path
        dl = ("  curl -L -o models/efficientdet_lite0.tflite \\\n"
              "    https://storage.googleapis.com/mediapipe-models/object_detector/"
              "efficientdet_lite0/float32/latest/efficientdet_lite0.tflite")
    else:
        model_path = cfg.model_path
        dl = ("  curl -L -o models/efficientdet_lite0_pp.tflite \\\n"
              "    https://storage.googleapis.com/download.tensorflow.org/models/tflite/"
              "task_library/object_detection/android/"
              "lite-model_efficientdet_lite0_detection_metadata_1.tflite")

    if not os.path.exists(model_path):
        print(f"ERROR: detector model not found at {model_path}\nDownload it with:\n{dl}",
              file=sys.stderr)
        return 1

    if cfg.llm_enabled and not os.environ.get("OPENAI_API_KEY"):
        print("note: OPENAI_API_KEY not set -> scene reasoning disabled "
              "(reactive loop unaffected).")
        cfg.llm_enabled = False

    if cfg.detector_backend == "mediapipe":
        from perception.detector import ObjectDetector
        detector = ObjectDetector(model_path, cfg.score_threshold, cfg.max_results,
                                  cfg.detect_size, cfg.allowed_labels)
    else:
        from perception.litert_detector import LiteRTDetector
        detector = LiteRTDetector(model_path, cfg.score_threshold, cfg.max_results,
                                  cfg.allowed_labels, cfg.num_threads)
    print(f"detector backend: {cfg.detector_backend} ({model_path})")
    async_det = AsyncDetector(detector)   # detection on its own thread
    async_det.start()
    tracker = Tracker(cfg.iou_match_threshold, cfg.max_age, cfg.min_hits)
    depth = MonoLoomingDepth()
    model_f = get_model(cfg.motion_model, cfg.drag_coeff)
    smoother = avoidance.CommandSmoother(cfg.command_hold_s)

    haptic = None
    if cfg.servo_enabled:
        from planning.haptic import HapticServo
        haptic = HapticServo(cfg.servo_pin, cfg.risk_threshold, cfg.servo_tap_deg,
                             cfg.servo_min_rate, cfg.servo_max_rate)
        haptic.start()

    # Always construct (it degrades to a no-op if the SDK/key is missing); only
    # spin up the background thread when LLM reasoning is enabled.
    scene = SceneReasoner(cfg.llm_model, cfg.llm_interval_s,
                          cfg.llm_jpeg_quality, cfg.llm_max_tokens)
    if cfg.llm_enabled:
        scene.start()

    streamer = None
    if cfg.headless:
        from viz.mjpeg import AnnotatedMJPEGServer
        streamer = AnnotatedMJPEGServer(cfg.stream_port, cfg.stream_jpeg_quality)
        streamer.start()
        print(f"serving annotated stream on http://<host>:{cfg.stream_port}")

    cam = build_camera(cfg)

    prev_ts = 0.0
    fps = 0.0
    last_action = None
    last_seq = -1
    # rolling per-stage timers (seconds) for --profile
    prof = {"read": 0.0, "phys": 0.0, "render": 0.0, "n": 0, "t0": time.time()}
    print("running. press 'q' in the window to quit (or Ctrl-C if headless).")

    try:
        while True:
            t_a = time.time()
            frame, ts = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue
            if ts == prev_ts:          # no fresh frame yet
                time.sleep(0.002)
                continue
            dt = (ts - prev_ts) if prev_ts else 1.0 / 30.0
            prev_ts = ts
            h, w = frame.shape[:2]

            # --- perception: detection runs on its own thread; we consume the
            #     latest results and coast on the Kalman prediction in between ---
            async_det.set_frame(frame)
            dets, seq = async_det.get()
            fresh = seq != last_seq
            last_seq = seq
            t_b = time.time()

            # --- physics + risk + planning (every frame) ---
            tracks = tracker.update(dets if fresh else [], dt)
            items, risks = [], []
            for t in tracks:
                traj = rk4.rollout(t.state, cfg.predict_horizon_s, cfg.predict_steps, model_f)
                _ = depth.estimate(t.state)   # range cue (relative); reserved for fusion
                r = collision.evaluate(t.state, traj, w, h, cfg.corridor_frac,
                                       cfg.ttc_warn_s, cfg.ttc_stop_s)
                risks.append(r)
                items.append({"track": t, "risk": r, "traj": traj})

            command = smoother.update(
                avoidance.decide(risks, cfg.risk_threshold, cfg.ttc_stop_s), ts)
            if command.action != last_action:
                print(f"[{time.strftime('%H:%M:%S')}] {command.action:5s} | {command.reason}")
                last_action = command.action

            if haptic is not None:
                worst = max((r.risk for r in risks), default=0.0)
                haptic.set_command(command, worst)

            scene.set_frame(frame)   # hand newest raw frame to the async scene model
            t_c = time.time()

            # --- render ---
            overlay.annotate(frame, items, command, scene.get(), fps,
                             cfg.corridor_frac, async_det.fps)
            inst = 1.0 / dt if dt > 0 else 0.0
            fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst

            if streamer is not None:
                streamer.set_frame(frame)
            else:
                cv2.imshow("vision_modal", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            t_d = time.time()

            if cfg.profile:
                prof["read"] += t_b - t_a
                prof["phys"] += t_c - t_b
                prof["render"] += t_d - t_c
                prof["n"] += 1
                if t_d - prof["t0"] >= 2.0:
                    n = max(prof["n"], 1)
                    print(f"[profile] render {fps:4.1f}fps | det {async_det.fps:4.1f}fps "
                          f"({async_det.infer_ms:5.1f}ms) | read {prof['read']/n*1e3:4.1f} "
                          f"phys {prof['phys']/n*1e3:4.1f} render {prof['render']/n*1e3:4.1f} ms")
                    prof.update(read=0.0, phys=0.0, render=0.0, n=0, t0=t_d)
    except KeyboardInterrupt:
        pass
    finally:
        scene.stop()
        if haptic is not None:
            haptic.stop()
        if streamer is not None:
            streamer.stop()
        async_det.stop()       # also closes the detector
        cam.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

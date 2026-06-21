"""Minimal MJPEG server for headless (Pi) viewing.

The app pushes annotated BGR frames in via `set_frame`; browsers at
http://<host>:<port> see the live annotated stream. Same wire format as stream.py,
but fed by the pipeline instead of the camera directly.
"""
from __future__ import annotations

import threading
from http import server
from socketserver import ThreadingMixIn

import cv2

_PAGE = b"""<!doctype html><html><head><title>vision_modal</title></head>
<body style="margin:0;background:#000">
<img src="stream.mjpg" style="width:100%;height:auto"/></body></html>"""


class _Server(ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class AnnotatedMJPEGServer:
    def __init__(self, port: int = 8000, jpeg_quality: int = 70):
        self._lock = threading.Condition()
        self._jpeg: bytes = b""
        self.jpeg_quality = jpeg_quality
        outer = self

        class Handler(server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence access log
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(_PAGE)))
                    self.end_headers()
                    self.wfile.write(_PAGE)
                elif self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=FRAME")
                    self.end_headers()
                    try:
                        while True:
                            with outer._lock:
                                outer._lock.wait()
                                frame = outer._jpeg
                            if not frame:
                                continue
                            self.wfile.write(b"--FRAME\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(frame)))
                            self.end_headers()
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                    except Exception:
                        pass
                else:
                    self.send_error(404)
                    self.end_headers()

        self._httpd = _Server(("", port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_frame(self, frame_bgr) -> None:
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
            self._lock.notify_all()

    def stop(self) -> None:
        self._httpd.shutdown()

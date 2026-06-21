"""Haptic servo controller — turns avoidance risk into a tactile "stimulation".

Runs on its OWN thread (like the camera/detector) so servo timing never blocks the
vision loop. Pattern: pulse-on-hazard — when a command is active, the servo gives
small taps; tap rate and amplitude scale with risk. When CLEAR, the servo detaches
(PWM off) so it sits still and doesn't buzz or hold torque.

All on the Raspberry Pi: signal from one GPIO pin (default GPIO18), servo V+/GND
from the Pi 5V/GND pins. gpiozero is imported lazily so the app still runs with
--servo off or on a laptop with no GPIO.
"""
from __future__ import annotations

import threading
import time

from .avoidance import Command


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


class HapticServo:
    def __init__(self, pin: int = 18, risk_threshold: float = 0.5,
                 tap_deg: float = 15.0, min_rate: float = 2.0, max_rate: float = 7.0,
                 min_pulse_s: float = 0.0005, max_pulse_s: float = 0.0025,
                 dwell_s: float = 0.05):
        self.risk_threshold = risk_threshold
        self.tap_deg = tap_deg
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.dwell_s = dwell_s

        self._active = False
        self._risk = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.available = False
        self._servo = None
        self._err = ""

        try:
            from gpiozero import AngularServo
            self._servo = AngularServo(
                pin, min_angle=-45, max_angle=45,
                min_pulse_width=min_pulse_s, max_pulse_width=max_pulse_s,
                initial_angle=0,
            )
            self.available = True
        except Exception as e:  # noqa: BLE001 - no GPIO / not on a Pi / pin busy
            self._err = str(e)

    def set_command(self, command: Command, risk: float) -> None:
        with self._lock:
            self._active = command.action != "CLEAR"
            self._risk = float(risk)

    def start(self) -> None:
        if not self.available:
            print(f"haptic servo disabled: {self._err or 'gpiozero unavailable'}")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("haptic servo active")

    def _tap(self, amp_deg: float) -> None:
        # one tap: centre -> +amp -> centre, then cut PWM so it rests
        self._servo.angle = amp_deg
        time.sleep(self.dwell_s)
        self._servo.angle = 0.0
        time.sleep(self.dwell_s)
        self._servo.detach()

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                active, risk = self._active, self._risk
            if not active or risk < self.risk_threshold:
                self._servo.detach()
                time.sleep(0.04)
                continue
            amp = self.tap_deg * _lerp(0.4, 1.0, risk)   # amplitude grows with risk
            rate = _lerp(self.min_rate, self.max_rate, risk)  # taps/sec grows with risk
            self._tap(amp)
            time.sleep(max(0.0, 1.0 / rate - 2 * self.dwell_s))

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._servo is not None:
            try:
                self._servo.detach()
                self._servo.close()
            except Exception:  # noqa: BLE001
                pass

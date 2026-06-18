"""
gesture_detector.py
--------------------
Hand-tracking and SOS-gesture recognition logic, built on MediaPipe's
HandLandmarker (the current Tasks API — the older `mediapipe.solutions.hands`
API has been removed from recent MediaPipe releases, so this is the
forward-compatible way to do it).

Two gestures are recognised, either of which can be "armed" independently:

1. SIGNAL_FOR_HELP - the real-world "Signal for Help" gesture popularised by
   the Canadian Women's Foundation: an open palm facing the camera with the
   thumb folded in across the palm, followed (within a short window) by the
   fingers closing down over the thumb into a fist. Because it requires two
   distinct stages in sequence, it is naturally resistant to accidental
   triggers from an ordinary raised hand or a casual fist.

2. FIST_HOLD - a plain closed fist held steady for N seconds. Simpler and
   very robust to detect, at the cost of being a less "symbolic" gesture.

NOTE ON THRESHOLDS: the finger/thumb classification below uses simple
geometric heuristics on MediaPipe's 21 hand landmarks. They work well for a
hand held upright and facing the camera at a normal distance, which is the
expected use case here, but lighting, camera angle, and landmark noise can
shift things. Tunable constants are grouped near the top of this file -
recalibrate them if you see misfires during testing.

MODEL FILE: HandLandmarker needs a small (~10MB) model bundle that isn't
shipped inside the mediapipe pip package. On first run this module downloads
it automatically from Google's public model store into ./models/. If your
machine has no internet access, download it manually from MODEL_URL and
place it at MODEL_PATH.
"""

import math
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# ---------------------------------------------------------------------------
# Model download (one-time)
# ---------------------------------------------------------------------------
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")


def ensure_model_downloaded(model_path: str = MODEL_PATH, url: str = MODEL_URL) -> str:
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, model_path)
    except Exception as exc:  # noqa: BLE001 - want a clear actionable message
        raise RuntimeError(
            "Could not automatically download the MediaPipe hand-landmark "
            f"model ({exc}). If this machine has no internet access, "
            f"manually download it from:\n  {url}\nand save it to:\n  {model_path}\n"
            "then restart the app."
        ) from exc
    return model_path


# ---------------------------------------------------------------------------
# Tunable constants for gesture classification
# ---------------------------------------------------------------------------
THUMB_TUCK_RATIO = 0.85   # thumb tip must be closer to the pinky-side of the
                          # palm than this fraction of the palm width to count
                          # as "tucked in" (used for Signal-for-Help stage 1)
MIN_CURLED_FOR_FIST = 3   # how many of the 4 main fingers must be curled to
                          # call the hand a fist

# Landmark indices (MediaPipe Hands, 21-point model)
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def classify_hand(landmarks) -> dict:
    """Turn raw hand landmarks (list of 21 objects with .x/.y) into the
    small set of booleans the gesture state machine cares about."""

    def curled(tip_idx, pip_idx) -> bool:
        # In image coordinates y grows downward, so a curled finger has its
        # tip BELOW (greater y than) its own pip joint.
        return landmarks[tip_idx].y > landmarks[pip_idx].y

    curls = [
        curled(INDEX_TIP, INDEX_PIP),
        curled(MIDDLE_TIP, MIDDLE_PIP),
        curled(RING_TIP, RING_PIP),
        curled(PINKY_TIP, PINKY_PIP),
    ]
    curled_count = sum(curls)

    palm_width = _dist(landmarks[INDEX_MCP], landmarks[PINKY_MCP])
    thumb_to_pinky_side = _dist(landmarks[THUMB_TIP], landmarks[PINKY_MCP])
    thumb_tucked = palm_width > 1e-6 and thumb_to_pinky_side < palm_width * THUMB_TUCK_RATIO

    is_fist = curled_count >= MIN_CURLED_FOR_FIST
    is_open_palm_thumb_tucked = curled_count == 0 and thumb_tucked

    return {
        "curled_count": curled_count,
        "thumb_tucked": thumb_tucked,
        "is_fist": is_fist,
        "is_open_palm_thumb_tucked": is_open_palm_thumb_tucked,
    }


def _draw_landmarks(frame_rgb: np.ndarray, landmarks) -> None:
    h, w = frame_rgb.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for conn in HAND_CONNECTIONS:
        cv2.line(frame_rgb, pts[conn.start], pts[conn.end], (0, 200, 80), 2)
    for (x, y) in pts:
        cv2.circle(frame_rgb, (x, y), 4, (255, 140, 0), -1)


# ---------------------------------------------------------------------------
# State machine for the two gestures
# ---------------------------------------------------------------------------
@dataclass
class GestureConfig:
    fist_hold_seconds: float = 3.0
    stage1_min_seconds: float = 0.4      # how long stage-1 pose must be held
    stage2_window_seconds: float = 3.0   # window to complete the fist after stage 1
    cooldown_seconds: float = 20.0       # minimum gap between two triggers


@dataclass
class _SOSState:
    fist_start: Optional[float] = None
    stage1_start: Optional[float] = None
    awaiting_stage2_until: Optional[float] = None
    last_trigger_time: Optional[float] = None


class SOSGestureEngine:
    """Wraps MediaPipe's HandLandmarker + the dual gesture state machine,
    frame by frame."""

    def __init__(self, config: Optional[GestureConfig] = None):
        self.config = config or GestureConfig()
        self.state = _SOSState()

        model_path = ensure_model_downloaded()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def close(self):
        self.landmarker.close()

    def in_cooldown(self, now: float) -> bool:
        lt = self.state.last_trigger_time
        return lt is not None and (now - lt) < self.config.cooldown_seconds

    def cooldown_remaining(self, now: float) -> float:
        if self.state.last_trigger_time is None:
            return 0.0
        return max(0.0, self.config.cooldown_seconds - (now - self.state.last_trigger_time))

    def process(self, frame_bgr: np.ndarray, armed: set, draw: bool = True):
        """
        Run one frame through detection + the gesture state machine.

        armed: a set containing any of {"signal_for_help", "fist_hold"}

        Returns (annotated_frame_rgb, event, status_text)
        event is None or one of {"signal_for_help", "fist_hold"}
        """
        now = time.time()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect(mp_image)

        event = None
        status_text = "No hand detected"

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            cls = classify_hand(landmarks)

            if draw:
                _draw_landmarks(frame_rgb, landmarks)

            cooling_down = self.in_cooldown(now)

            # --- fist_hold gesture -------------------------------------------------
            if "fist_hold" in armed:
                if cls["is_fist"]:
                    if self.state.fist_start is None:
                        self.state.fist_start = now
                    held = now - self.state.fist_start
                    if held >= self.config.fist_hold_seconds and not cooling_down:
                        event = "fist_hold"
                        self.state.last_trigger_time = now
                        self.state.fist_start = None
                    else:
                        status_text = f"Fist held {held:.1f}s / {self.config.fist_hold_seconds:.0f}s"
                else:
                    self.state.fist_start = None

            # --- signal_for_help gesture --------------------------------------------
            if "signal_for_help" in armed and event is None:
                if self.state.awaiting_stage2_until is not None:
                    if now <= self.state.awaiting_stage2_until:
                        if cls["is_fist"] and not cooling_down:
                            event = "signal_for_help"
                            self.state.last_trigger_time = now
                            self.state.awaiting_stage2_until = None
                            self.state.stage1_start = None
                        else:
                            remaining = self.state.awaiting_stage2_until - now
                            status_text = f"Stage 1 done — close fingers over thumb now! ({remaining:.1f}s left)"
                    else:
                        self.state.awaiting_stage2_until = None
                else:
                    if cls["is_open_palm_thumb_tucked"]:
                        if self.state.stage1_start is None:
                            self.state.stage1_start = now
                        held1 = now - self.state.stage1_start
                        if held1 >= self.config.stage1_min_seconds:
                            self.state.awaiting_stage2_until = now + self.config.stage2_window_seconds
                            self.state.stage1_start = None
                            status_text = "Stage 1 confirmed — now close your fingers over your thumb"
                        else:
                            status_text = f"Stage 1 holding ({held1:.1f}s)..."
                    else:
                        self.state.stage1_start = None

            if cooling_down and event is None:
                status_text = f"Cooldown ({self.cooldown_remaining(now):.0f}s)"
            elif event is None and status_text == "No hand detected":
                status_text = "Hand detected — no gesture in progress"
        else:
            # hand lost — reset partial progress so it can't be "carried over"
            self.state.fist_start = None
            self.state.stage1_start = None
            self.state.awaiting_stage2_until = None

        return frame_rgb, event, status_text

"""
emotion_detector.py — Facial Landmark-Based Emotion Detection
=============================================================
Analyzes MediaPipe face mesh landmarks to infer emotional state.
Used to modulate TTS pitch/rate/volume accordingly.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EmotionState:
    label:      str    = "neutral"
    confidence: float  = 0.0
    tts_rate:   int    = 155
    tts_volume: float  = 0.85
    tts_pitch:  float  = 1.0


# ─── Key facial landmark indices (MediaPipe Face Mesh 468) ───
#
# Eye openness:  upper_lid vs lower_lid
# Mouth openness: upper_lip vs lower_lip
# Eyebrow raise: eyebrow vs eye

# LEFT EYE
LEFT_EYE_UPPER  = [159, 158, 157, 173]
LEFT_EYE_LOWER  = [145, 153, 154, 155]
LEFT_IRIS       = 468   # Requires refine_face_landmarks=True

# RIGHT EYE
RIGHT_EYE_UPPER = [386, 385, 384, 398]
RIGHT_EYE_LOWER = [374, 380, 381, 382]

# MOUTH
MOUTH_UPPER     = [13, 312, 311, 310, 415]
MOUTH_LOWER     = [14, 317, 402, 318, 324]
MOUTH_LEFT      = 61
MOUTH_RIGHT     = 291

# EYEBROWS
LEFT_BROW       = [70, 63, 105, 66, 107]
RIGHT_BROW      = [336, 296, 334, 293, 300]

# ─────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────

def _landmark_arr(face_landmarks, indices: list) -> np.ndarray:
    """Extract (x, y, z) for the given landmark indices."""
    lms = face_landmarks.landmark
    return np.array([[lms[i].x, lms[i].y, lms[i].z] for i in indices])


def _vertical_distance(upper_pts: np.ndarray, lower_pts: np.ndarray) -> float:
    """Mean vertical (y-axis) distance between two landmark groups."""
    return float(np.mean(lower_pts[:, 1]) - np.mean(upper_pts[:, 1]))


def _horizontal_distance(pt_a, pt_b) -> float:
    return float(np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]))


# ─────────────────────────────────────────────────────────────
# METRICS COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_facial_metrics(face_landmarks) -> dict:
    """
    Return a dict of normalised facial metrics in [0, 1].
    """
    lms = face_landmarks.landmark

    def pt(idx):
        return np.array([lms[idx].x, lms[idx].y, lms[idx].z])

    # Eye Aspect Ratio (EAR) — how open are the eyes?
    def ear(upper_idx, lower_idx):
        up  = _landmark_arr(face_landmarks, upper_idx)
        low = _landmark_arr(face_landmarks, lower_idx)
        vertical = _vertical_distance(up, low)
        return max(vertical, 0.0)

    left_ear  = ear(LEFT_EYE_UPPER,  LEFT_EYE_LOWER)
    right_ear = ear(RIGHT_EYE_UPPER, RIGHT_EYE_LOWER)
    avg_ear   = (left_ear + right_ear) / 2.0

    # Mouth Aspect Ratio (MAR)
    mouth_up  = _landmark_arr(face_landmarks, MOUTH_UPPER)
    mouth_low = _landmark_arr(face_landmarks, MOUTH_LOWER)
    mouth_vertical   = _vertical_distance(mouth_up, mouth_low)
    mouth_horizontal = _horizontal_distance(pt(MOUTH_LEFT), pt(MOUTH_RIGHT))
    mar = mouth_vertical / (mouth_horizontal + 1e-6)

    # Eyebrow Raise Ratio — brow y vs eye y
    def brow_raise(brow_idx, eye_upper_idx):
        brow = _landmark_arr(face_landmarks, brow_idx)
        eye  = _landmark_arr(face_landmarks, eye_upper_idx)
        return float(np.mean(eye[:, 1]) - np.mean(brow[:, 1]))  # positive → raised

    left_brow_raise  = brow_raise(LEFT_BROW,  LEFT_EYE_UPPER)
    right_brow_raise = brow_raise(RIGHT_BROW, RIGHT_EYE_UPPER)
    avg_brow_raise   = (left_brow_raise + right_brow_raise) / 2.0

    # Normalise to [0, 1] using empirical bounds
    ear_norm   = np.clip(avg_ear        / 0.05, 0, 1)
    mar_norm   = np.clip(mar            / 0.40, 0, 1)
    brow_norm  = np.clip(avg_brow_raise / 0.05, 0, 1)

    return {
        "ear":          avg_ear,
        "ear_norm":     float(ear_norm),
        "mar":          mar,
        "mar_norm":     float(mar_norm),
        "brow_raise":   avg_brow_raise,
        "brow_norm":    float(brow_norm),
    }


# ─────────────────────────────────────────────────────────────
# RULE-BASED EMOTION CLASSIFICATION
# ─────────────────────────────────────────────────────────────

EMOTION_RULES = {
    # (ear_norm_max, mar_norm_threshold, brow_raise_min)
    "angry":       {"ear_max": 0.35, "mar_min": 0.20, "brow_min": -1.0,
                    "rate": 190, "volume": 1.0,  "pitch": 0.85},
    "happy":       {"ear_max": 1.0,  "mar_min": 0.35, "brow_min": 0.40,
                    "rate": 170, "volume": 0.90, "pitch": 1.1},
    "questioning": {"ear_max": 1.0,  "mar_min": 0.15, "brow_min": 0.60,
                    "rate": 140, "volume": 0.75, "pitch": 1.15},
    "sad":         {"ear_max": 0.30, "mar_min": 0.05, "brow_min": -1.0,
                    "rate": 130, "volume": 0.70, "pitch": 0.95},
    "neutral":     {"ear_max": 1.0,  "mar_min": 0.00, "brow_min": -1.0,
                    "rate": 155, "volume": 0.85, "pitch": 1.0},
}


class EmotionDetector:
    """
    Stateful emotion detector with sliding-window smoothing.
    """

    def __init__(self, smooth_window: int = 8):
        self._window     = smooth_window
        self._history    = []   # Stores recent emotion labels
        self._current    = EmotionState()

    def update(self, face_landmarks) -> EmotionState:
        """
        Feed the latest face landmarks, returns the smoothed EmotionState.
        """
        if face_landmarks is None:
            return self._current

        try:
            metrics = compute_facial_metrics(face_landmarks)
            label   = self._classify(metrics)
            confidence = self._update_history(label)
            params  = EMOTION_RULES.get(label, EMOTION_RULES["neutral"])

            self._current = EmotionState(
                label      = label,
                confidence = confidence,
                tts_rate   = params["rate"],
                tts_volume = params["volume"],
                tts_pitch  = params["pitch"],
            )
        except Exception:
            pass   # Don't crash the inference loop on bad frames

        return self._current

    def _classify(self, metrics: dict) -> str:
        ear_n  = metrics["ear_norm"]
        mar_n  = metrics["mar_norm"]
        brow_n = metrics["brow_norm"]

        scores = {}
        for emotion, rule in EMOTION_RULES.items():
            satisfied = (
                ear_n  <= rule["ear_max"]  and
                mar_n  >= rule["mar_min"]  and
                brow_n >= rule["brow_min"]
            )
            scores[emotion] = 1.0 if satisfied else 0.0

        # Priority order for tie-breaking
        for emo in ["angry", "happy", "questioning", "sad", "neutral"]:
            if scores.get(emo, 0) > 0:
                return emo
        return "neutral"

    def _update_history(self, label: str) -> float:
        self._history.append(label)
        if len(self._history) > self._window:
            self._history.pop(0)
        count = self._history.count(label)
        return count / len(self._history)

    @property
    def current_state(self) -> EmotionState:
        return self._current

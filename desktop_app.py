"""
app.py — Real-Time Multi-Modal Sign Language Recognition
=========================================================
Main application entry point.

Architecture (Antigravity Framework):
  ┌─────────────────────────────────────────────────────────┐
  │  MainThread        — Camera capture + OpenCV UI display │
  │  PredictionAsync   — Bi-LSTM inference worker           │
  │  SpeechDaemon      — TTS output (never blocks UI)       │
  └─────────────────────────────────────────────────────────┘

Controls:
  [1] Currently only ISL is supported.
  [S] Toggle speech mute/unmute
  [R] Reset text buffer
  [Q] or [ESC] Quit
"""

import os
import sys
import cv2
import time
import logging
import threading
import numpy as np
import mediapipe as mp
from collections import deque

# Local modules
from config import (
    MP_CONFIG, TOTAL_FEATURES, UI, LANGUAGES,
    POSE_FEATURES, FACE_FEATURES, LH_FEATURES, RH_FEATURES,
    LOGS_DIR,
)
from emotion_detector import EmotionDetector
from tts_engine       import TTSEngine
from inference_engine import InferenceEngine

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "app.log")),
    ]
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# MEDIAPIPE EXTRACTION HELPER
# ─────────────────────────────────────────────────────────────

def extract_keypoints(results) -> np.ndarray:
    pose = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(POSE_FEATURES))

    face = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks else np.zeros(FACE_FEATURES))

    lh   = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks else np.zeros(LH_FEATURES))

    rh   = (np.array([[lm.x, lm.y, lm.z]
                      for lm in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks else np.zeros(RH_FEATURES))

    return np.concatenate([pose, face, lh, rh])


# ─────────────────────────────────────────────────────────────
# SKELETON DRAWING
# ─────────────────────────────────────────────────────────────

def draw_styled_landmarks(image, results, mp_drawing, mp_holistic, mp_drawing_styles):
    """Draw connected skeleton with custom colors and glow effect."""

    # ── Face Mesh ────────────────────────────────────────────
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style(),
        )

    # ── Pose ─────────────────────────────────────────────────
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=UI["skeleton_color"], thickness=2, circle_radius=3
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 180, 120), thickness=2
            ),
        )

    # ── Hands ─────────────────────────────────────────────────
    for hand_landmarks, connections in [
        (results.left_hand_landmarks,  mp_holistic.HAND_CONNECTIONS),
        (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS),
    ]:
        if hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                connections,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 100, 230), thickness=2, circle_radius=4
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(200, 50, 200), thickness=2
                ),
            )


# ─────────────────────────────────────────────────────────────
# UI OVERLAY DRAWING
# ─────────────────────────────────────────────────────────────

def draw_overlay(
    image, lang, muted, emotion, fill_ratio,
    prediction_conf, last_word,
    buffered_text, last_sentence,
    fps,
):
    """Render all HUD elements onto the frame."""
    h, w = image.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    WHITE = UI["text_color"]
    CYAN  = UI["accent_color"]
    GREEN = UI["skeleton_color"]
    GOLD  = UI["fps_color"]

    # ── Top-left: Language badge ──────────────────────────────
    _draw_badge(image, f"  {lang}  ", (10, 10), (80, 50),
                color=(30, 120, 220))

    # ── FPS counter ───────────────────────────────────────────
    cv2.putText(image, f"FPS: {fps:.0f}", (10, 75),
                font, 0.55, GOLD, 1, cv2.LINE_AA)

    # ── Speech status ─────────────────────────────────────────
    mic_label = "  MUTED  " if muted else " SPEECH  "
    mic_color = (60, 60, 60) if muted else (30, 160, 50)
    _draw_badge(image, mic_label, (w - 120, 10), (w - 10, 50), mic_color)

    # ── Emotion badge ─────────────────────────────────────────
    emo_colors = {
        "happy":       (0, 180, 80),
        "angry":       (0, 40, 220),
        "questioning": (200, 140, 0),
        "sad":         (120, 50, 0),
        "neutral":     (60, 60, 60),
    }
    emo_c = emo_colors.get(emotion.label, (60, 60, 60))
    _draw_badge(image, f"  {emotion.label.upper()}  ",
                (w - 180, 60), (w - 10, 100), emo_c)

    # ── Capture progress bar ──────────────────────────────────
    bar_y  = h - 220
    bar_w  = int((w - 40) * fill_ratio)
    bar_bg = (30, 30, 30)
    bar_fg = (0, 180, 255) if fill_ratio < 1 else GREEN
    cv2.rectangle(image, (20, bar_y), (w - 20, bar_y + 10), bar_bg,  -1)
    cv2.rectangle(image, (20, bar_y), (20 + bar_w, bar_y + 10), bar_fg, -1)
    cv2.putText(image, f"Capture: {fill_ratio*100:.0f}%",
                (20, bar_y - 5), font, 0.42, CYAN, 1, cv2.LINE_AA)

    # ── Confidence meter ──────────────────────────────────────
    conf_y   = bar_y + 20
    conf_w   = int((w - 40) * prediction_conf)
    conf_col = _confidence_color(prediction_conf)
    cv2.rectangle(image, (20, conf_y), (w - 20, conf_y + 10), bar_bg, -1)
    cv2.rectangle(image, (20, conf_y), (20 + conf_w, conf_y + 10), conf_col, -1)
    cv2.putText(image, f"Confidence: {prediction_conf*100:.1f}%",
                (20, conf_y - 5), font, 0.42, CYAN, 1, cv2.LINE_AA)

    # ── Current word ──────────────────────────────────────────
    if last_word:
        word_size = cv2.getTextSize(last_word, font, 1.4, 3)[0]
        wx = (w - word_size[0]) // 2
        # Drop shadow
        cv2.putText(image, last_word, (wx + 2, h - 145), font, 1.4,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, last_word, (wx, h - 147), font, 1.4,
                    GREEN, 3, cv2.LINE_AA)

    # ── Buffered partial sentence ─────────────────────────────
    _draw_transparent_banner(image, buffered_text, h - 110, w)

    # ── Last complete sentence ────────────────────────────────
    _draw_transparent_banner(image, f"\"{last_sentence}\"", h - 70, w,
                             bg=(0, 0, 0), text_color=GOLD)

    # ── Controls guide (bottom right) ────────────────────────
    guide = ["[1] Language", "[S] Mute", "[R] Reset", "[Q] Quit"]
    for i, g in enumerate(guide):
        cv2.putText(image, g, (w - 140, h - 90 + i * 22),
                    font, 0.42, (150, 150, 150), 1, cv2.LINE_AA)


def _draw_badge(image, text, pt1, pt2, color):
    overlay = image.copy()
    cv2.rectangle(overlay, pt1, pt2, color, -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv2.putText(image, text,
                (pt1[0] + 4, pt1[1] + (pt2[1] - pt1[1]) // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                UI["text_color"], 1, cv2.LINE_AA)


def _draw_transparent_banner(image, text, y, w, bg=(0, 0, 0),
                              text_color=(255, 255, 255)):
    if not text:
        return
    overlay = image.copy()
    cv2.rectangle(overlay, (0, y), (w, y + 36), bg, -1)
    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

    # Word-wrap text that is too long
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 1
    max_w = w - 40
    chars_per_line = max(1, max_w // 12)
    display = text[:chars_per_line] + ("…" if len(text) > chars_per_line else "")

    cv2.putText(image, display, (20, y + 24),
                font, scale, text_color, thick, cv2.LINE_AA)


def _confidence_color(conf: float) -> tuple:
    """Green for high confidence, red for low."""
    r = int(255 * (1 - conf))
    g = int(255 * conf)
    return (0, g, r)


# ─────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────

LANG_CYCLE = ["ISL"]


def main():
    log.info("=" * 65)
    log.info("  SLR Studio — Multi-Modal Sign Language Recognition")
    log.info("  ISL   ·   Real-Time Inference")
    log.info("=" * 65)

    # ── MediaPipe setup ───────────────────────────────────────
    mp_holistic       = mp.solutions.holistic
    mp_drawing        = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    holistic = mp_holistic.Holistic(**MP_CONFIG)

    # ── Sub-systems ───────────────────────────────────────────
    emotion_detector = EmotionDetector(smooth_window=10)

    tts_backend = "pyttsx3"   # Change to "gtts" for online mode
    tts_engine  = TTSEngine(backend=tts_backend, max_queue=5)
    tts_engine.start()

    lang_idx           = 0
    current_lang       = LANG_CYCLE[lang_idx]
    inference_engine   = InferenceEngine(lang=current_lang)
    inference_engine.start()

    # ── State ─────────────────────────────────────────────────
    last_word          = ""
    buffered_text      = ""
    last_sentence      = ""
    prediction_conf    = 0.0
    fill_ratio         = 0.0

    fps_history        = deque(maxlen=30)
    prev_time          = time.time()

    # ── Camera ────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  UI["window_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, UI["window_height"])
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        log.error("Cannot open webcam. Check device index.")
        return

    cv2.namedWindow(UI["window_title"], cv2.WINDOW_NORMAL)
    cv2.resizeWindow(UI["window_title"],
                     UI["window_width"], UI["window_height"])

    log.info("Camera started. Press [Q] or [ESC] to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Empty frame — retrying …")
                continue

            # ── FPS ───────────────────────────────────────────
            now  = time.time()
            fps_history.append(1.0 / max(now - prev_time, 1e-6))
            fps  = float(np.mean(fps_history))
            prev_time = now

            # ── MediaPipe processing ──────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # ── Skeleton overlay ──────────────────────────────
            draw_styled_landmarks(
                frame, results, mp_drawing, mp_holistic, mp_drawing_styles
            )

            # ── Emotion detection ─────────────────────────────
            emotion = emotion_detector.update(results.face_landmarks)

            # ── Landmark extraction → Inference queue ─────────
            kp = extract_keypoints(results)
            inference_engine.push_frame(kp)
            fill_ratio = inference_engine.fill_ratio

            # ── Read inference result ─────────────────────────
            result = inference_engine.get_result()
            if result:
                if result.prediction and result.prediction.word:
                    last_word       = result.prediction.word
                    prediction_conf = result.prediction.confidence

                buffered_text = result.buffered_text

                if result.is_sentence_complete and result.buffered_text:
                    last_sentence = result.buffered_text
                    lang_cfg      = LANGUAGES[current_lang]
                    # Speak the complete sentence with emotion modulation
                    tts_engine.speak_with_emotion(
                        last_sentence, emotion,
                        lang=lang_cfg["tts_lang"],
                    )

            # ── HUD overlay ───────────────────────────────────
            draw_overlay(
                frame,
                lang=current_lang,
                muted=tts_engine.is_muted,
                emotion=emotion,
                fill_ratio=fill_ratio,
                prediction_conf=prediction_conf,
                last_word=last_word,
                buffered_text=buffered_text,
                last_sentence=last_sentence,
                fps=fps,
            )

            # ── Display ───────────────────────────────────────
            cv2.imshow(UI["window_title"], frame)

            # ── Key handling ──────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):   # Q or ESC
                log.info("Quit key pressed.")
                break

            elif key == ord("1"):       # Cycle language
                lang_idx     = (lang_idx + 1) % len(LANG_CYCLE)
                current_lang = LANG_CYCLE[lang_idx]
                inference_engine.set_language(current_lang)
                last_word       = ""
                buffered_text   = ""
                prediction_conf = 0.0
                log.info(f"Language switched to {current_lang}.")

            elif key == ord("s"):       # Mute toggle
                tts_engine.toggle_mute()

            elif key == ord("r"):       # Reset buffer
                buffered_text = ""
                last_word     = ""
                prediction_conf = 0.0
                log.info("Text buffer reset.")

    finally:
        # ── Cleanup ───────────────────────────────────────────
        log.info("Shutting down …")
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        tts_engine.stop()
        inference_engine.stop()
        log.info("SLR Studio closed cleanly.")


if __name__ == "__main__":
    main()

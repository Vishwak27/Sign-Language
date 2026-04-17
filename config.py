"""
config.py — Central Configuration for the Multi-Modal SLR System
================================================================
All paths, hyperparameters, and runtime flags in one place.
"""

import os

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR     = os.path.join(BASE_DIR, "datasets")
ISL_DATASET_DIR = os.path.join(DATASET_DIR, "Include")

LANDMARK_DIR    = os.path.join(BASE_DIR, "landmarks")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

# Auto-create directories
for _dir in [LANDMARK_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR,
             ISL_DATASET_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# MEDIAPIPE HOLISTIC SETTINGS
# ─────────────────────────────────────────────────────────────
MP_CONFIG = {
    "static_image_mode":        False,
    "model_complexity":         2,          # 0=light, 1=full, 2=heavy
    "smooth_landmarks":         True,       # Temporal smoothing
    "enable_segmentation":      False,
    "smooth_segmentation":      True,
    "refine_face_landmarks":    False,       # 478 vs 468 landmarks
    "min_detection_confidence": 0.60,
    "min_tracking_confidence":  0.60,
}

# ─────────────────────────────────────────────────────────────
# LANDMARK FEATURE DIMENSIONS
# ─────────────────────────────────────────────────────────────
POSE_LANDMARKS    = 33   # Upper body
FACE_LANDMARKS    = 468  # Full face mesh
LH_LANDMARKS      = 21   # Left hand
RH_LANDMARKS      = 21   # Right hand

# Each landmark has (x, y, z) → 3 values
POSE_FEATURES     = POSE_LANDMARKS * 3       # 99
FACE_FEATURES     = FACE_LANDMARKS * 3       # 1404
LH_FEATURES       = LH_LANDMARKS * 3        # 63
RH_FEATURES       = RH_LANDMARKS * 3        # 63
TOTAL_FEATURES    = POSE_FEATURES + FACE_FEATURES + LH_FEATURES + RH_FEATURES  # 1629

# ─────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────
SEQUENCE_LENGTH   = 30     # Frames per gesture sample
BATCH_SIZE        = 32
EPOCHS_BASE       = 50     # Base training
EPOCHS_FINETUNE   = 25     # Fine-tuning
LEARNING_RATE     = 1e-3
FINETUNE_LR       = 1e-4
DROPOUT_RATE      = 0.4
LSTM_UNITS        = [128, 64]   # Bi-LSTM layer sizes
DENSE_UNITS       = [64, 32]

# ─────────────────────────────────────────────────────────────
# LANGUAGE SETTINGS
# ─────────────────────────────────────────────────────────────
LANGUAGES = {
    "ISL": {
        "name":         "Indian Sign Language",
        "grammar":      "SOV",
        "tts_lang":     "hi",
        "tts_voice_id": None,
        "display_lang": "Hindi/English",
    },
}

# ─────────────────────────────────────────────────────────────
# INFERENCE ENGINE SETTINGS
# ─────────────────────────────────────────────────────────────
INFERENCE_THRESHOLD      = 0.75   # Min confidence to accept prediction
PAUSE_TIMEOUT_SECONDS    = 1.8    # Silence duration to flush text buffer
SPEECH_QUEUE_MAX         = 5      # Max items in speech queue
TEMPORAL_SMOOTH_FRAMES   = 5      # Frames for prediction smoothing

# ─────────────────────────────────────────────────────────────
# EMOTION / SENTIMENT DETECTION (Facial)
# ─────────────────────────────────────────────────────────────
EMOTION_TTS_PARAMS = {
    "happy":       {"rate": 170, "volume": 0.9,  "pitch": 1.1},
    "angry":       {"rate": 190, "volume": 1.0,  "pitch": 0.85},
    "questioning": {"rate": 140, "volume": 0.75, "pitch": 1.15},
    "neutral":     {"rate": 155, "volume": 0.85, "pitch": 1.0},
    "sad":         {"rate": 130, "volume": 0.7,  "pitch": 0.95},
}

# ─────────────────────────────────────────────────────────────
# UI SETTINGS
# ─────────────────────────────────────────────────────────────
UI = {
    "window_title":      "SLR Studio — Multi-Modal Sign Recognition",
    "window_width":      1280,
    "window_height":     720,
    "skeleton_color":    (0, 255, 180),   # Cyan-green
    "text_color":        (255, 255, 255),
    "accent_color":      (80, 200, 255),
    "fps_color":         (200, 200, 0),
    "confidence_bar_h":  22,
    "font_scale":        0.6,
    "font_thickness":    2,
    "overlay_alpha":     0.5,
}

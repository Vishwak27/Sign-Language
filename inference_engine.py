"""
inference_engine.py — Real-Time Gesture Classification Engine
=============================================================
Manages the prediction queue, temporal smoothing, and text buffer.
Designed to run in a high-priority async worker thread.
"""

import os
import json
import queue
import logging
import threading
import time
import numpy as np
from collections import deque
from typing import Optional
from dataclasses import dataclass, field

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras

from config import (
    MODELS_DIR, INFERENCE_THRESHOLD, PAUSE_TIMEOUT_SECONDS,
    SEQUENCE_LENGTH, TOTAL_FEATURES, TEMPORAL_SMOOTH_FRAMES,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class Prediction:
    word:       str    = ""
    confidence: float  = 0.0
    lang:       str    = "ISL"
    timestamp:  float  = field(default_factory=time.time)


@dataclass
class InferenceResult:
    prediction:     Optional[Prediction] = None
    buffered_text:  str                  = ""
    is_sentence_complete: bool           = False


# ─────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────

class ModelStore:
    """Loads and caches all three language models + class labels."""

    CONFIGS = {
        "ISL": ("isl_model.keras",      "isl_classes.json"),
    }

    def __init__(self):
        self._models:  dict[str, keras.Model] = {}
        self._classes: dict[str, list]        = {}
        self._lock     = threading.Lock()

    def load(self, lang: str) -> bool:
        """Load a model for a given language. Thread-safe."""
        with self._lock:
            if lang in self._models:
                return True

            model_file, class_file = self.CONFIGS[lang]
            model_path = os.path.join(MODELS_DIR, model_file)
            class_path = os.path.join(MODELS_DIR, class_file)

            if not os.path.exists(model_path):
                log.warning(f"[{lang}] Model not found: {model_path}")
                log.warning(f"         Run train_slr_models.py first.")
                return False

            if not os.path.exists(class_path):
                log.warning(f"[{lang}] Class file not found: {class_path}")
                return False

            try:
                model = keras.models.load_model(model_path)
                with open(class_path) as f:
                    classes = json.load(f)
                self._models[lang]  = model
                self._classes[lang] = classes
                log.info(f"[{lang}] Model loaded  ({len(classes)} classes).")
                return True
            except Exception as e:
                log.error(f"[{lang}] Failed to load model: {e}")
                return False

    def load_all(self):
        for lang in self.CONFIGS:
            self.load(lang)

    def predict(self, lang: str, sequence: np.ndarray) -> tuple[str, float]:
        """
        Run inference. Returns (word, confidence).
        sequence shape: (SEQUENCE_LENGTH, TOTAL_FEATURES)
        """
        if lang not in self._models:
            return "", 0.0

        model   = self._models[lang]
        classes = self._classes[lang]

        inp      = sequence[np.newaxis, ...]           # (1, 30, 1629)
        probs    = model.predict(inp, verbose=0)[0]    # (num_classes,)
        idx      = int(np.argmax(probs))
        conf     = float(probs[idx])
        word     = classes[idx] if idx < len(classes) else "?"
        return word, conf

    def is_loaded(self, lang: str) -> bool:
        return lang in self._models

    def num_classes(self, lang: str) -> int:
        return len(self._classes.get(lang, []))


# ─────────────────────────────────────────────────────────────
# LANDMARK FRAME BUFFER
# ─────────────────────────────────────────────────────────────

class FrameBuffer:
    """
    Collects extracted landmark frames until a full sequence is ready.
    Applies temporal smoothing using a sliding window average.
    """

    def __init__(self, seq_len: int = SEQUENCE_LENGTH,
                 smooth_window: int = TEMPORAL_SMOOTH_FRAMES):
        self._seq_len = seq_len
        self._smooth  = smooth_window
        self._buffer: deque = deque(maxlen=seq_len * 2)

    def push(self, frame: np.ndarray):
        """Push a single landmark vector (TOTAL_FEATURES,)."""
        self._buffer.append(frame)

    def ready(self) -> bool:
        return len(self._buffer) >= self._seq_len

    def get_sequence(self) -> np.ndarray:
        """
        Return the latest seq_len frames as a smoothed numpy array
        of shape (SEQUENCE_LENGTH, TOTAL_FEATURES).
        """
        frames = np.array(list(self._buffer)[-self._seq_len:])  # (30, 1629)

        # Temporal smoothing: running mean over adjacent frames
        smoothed = np.copy(frames)
        w = self._smooth
        for i in range(len(frames)):
            start = max(0, i - w // 2)
            end   = min(len(frames), i + w // 2 + 1)
            smoothed[i] = frames[start:end].mean(axis=0)

        return smoothed.astype(np.float32)

    def clear(self):
        self._buffer.clear()

    def fill_ratio(self) -> float:
        return min(len(self._buffer) / self._seq_len, 1.0)


# ─────────────────────────────────────────────────────────────
# PREDICTION SMOOTHER
# ─────────────────────────────────────────────────────────────

class PredictionSmoother:
    """
    Holds the last N predictions and returns the majority vote.
    Prevents single-frame false positives from reaching the text buffer.
    """

    def __init__(self, window: int = TEMPORAL_SMOOTH_FRAMES):
        self._window  = window
        self._history = deque(maxlen=window)

    def update(self, word: str, confidence: float) -> tuple[str, float]:
        if confidence >= INFERENCE_THRESHOLD:
            self._history.append((word, confidence))

        if not self._history:
            return "", 0.0

        # Majority vote
        from collections import Counter
        words    = [h[0] for h in self._history]
        votes    = Counter(words)
        top_word = votes.most_common(1)[0][0]
        confs    = [h[1] for h in self._history if h[0] == top_word]
        avg_conf = float(np.mean(confs))

        return top_word, avg_conf

    def reset(self):
        self._history.clear()


# ─────────────────────────────────────────────────────────────
# TEXT BUFFER (Buffer & Flush)
# ─────────────────────────────────────────────────────────────

class TextBuffer:
    """
    Accumulates recognized words and flushes a complete sentence
    when a pause is detected or a pause-gesture is recognized.
    """

    PAUSE_GESTURES = {"stop", "pause", "end", "period", "full_stop"}

    def __init__(self, pause_timeout: float = PAUSE_TIMEOUT_SECONDS):
        self._words:        list[str] = []
        self._last_word_ts: float     = 0.0
        self._pause_timeout           = pause_timeout
        self._last_sentence:          str = ""

    def push_word(self, word: str):
        """Add a recognized word. Ignores pause-gesture tokens."""
        if word.lower() in self.PAUSE_GESTURES:
            return
        self._words.append(word)
        self._last_word_ts = time.time()

    def should_flush(self) -> bool:
        if not self._words:
            return False
        return (time.time() - self._last_word_ts) >= self._pause_timeout

    def flush(self) -> str:
        """Return the accumulated sentence and clear the buffer."""
        sentence = " ".join(self._words).strip()
        self._words.clear()
        self._last_sentence = sentence
        return sentence

    @property
    def current_partial(self) -> str:
        return " ".join(self._words)

    @property
    def last_sentence(self) -> str:
        return self._last_sentence


# ─────────────────────────────────────────────────────────────
# INFERENCE ENGINE — MAIN CLASS
# ─────────────────────────────────────────────────────────────

class InferenceEngine:
    """
    High-priority async inference engine.

    Main thread pushes landmark frames via push_frame().
    A background worker thread drains the frame queue,
    runs predictions, and fills the result queue.

    Consumer reads from get_result() on main loop tick.
    """

    def __init__(self, lang: str = "ISL"):
        self._lang        = lang
        self._store       = ModelStore()
        self._frame_buf   = FrameBuffer()
        self._smoother    = PredictionSmoother()
        self._text_buf    = TextBuffer()
        self._frame_q: queue.Queue = queue.Queue(maxsize=10)
        self._result_q: queue.Queue = queue.Queue(maxsize=20)
        self._running     = False
        self._thread:     Optional[threading.Thread] = None
        self._last_pred   = Prediction()

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self):
        self._store.load(self._lang)
        self._running = True
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="PredictionAsyncWorker",
        )
        self._thread.start()
        log.info(f"InferenceEngine started (lang={self._lang}).")

    def stop(self):
        self._running = False
        log.info("InferenceEngine stopped.")

    # ── Public API ────────────────────────────────────────────

    def set_language(self, lang: str):
        """Hot-swap the active sign language."""
        if lang == self._lang:
            return
        self._lang = lang
        self._store.load(lang)
        self._frame_buf.clear()
        self._smoother.reset()
        log.info(f"Language switched to {lang}.")

    def push_frame(self, landmark_vector: np.ndarray):
        """
        Push a single frame's landmark vector (shape: (TOTAL_FEATURES,)).
        Non-blocking — drops frame if worker is behind.
        """
        self._frame_buf.push(landmark_vector)

        if self._frame_buf.ready():
            seq = self._frame_buf.get_sequence()
            try:
                self._frame_q.put_nowait(seq)
            except queue.Full:
                pass   # Inference worker is behind — skip frame

    def get_result(self) -> Optional[InferenceResult]:
        """
        Return the latest InferenceResult (non-blocking).
        Returns None if no new prediction is available.
        """
        # Check if current buffer should flush
        if self._text_buf.should_flush():
            sentence = self._text_buf.flush()
            return InferenceResult(
                prediction=self._last_pred,
                buffered_text=sentence,
                is_sentence_complete=True,
            )

        try:
            result = self._result_q.get_nowait()
            return result
        except queue.Empty:
            return InferenceResult(
                buffered_text=self._text_buf.current_partial,
            )

    @property
    def fill_ratio(self) -> float:
        """0→1 showing how full the frame capture buffer is."""
        return self._frame_buf.fill_ratio()

    @property
    def current_lang(self) -> str:
        return self._lang

    @property
    def model_loaded(self) -> bool:
        return self._store.is_loaded(self._lang)

    # ── Background Worker ─────────────────────────────────────

    def _worker(self):
        """High-priority prediction worker (runs in dedicated thread)."""
        log.info("Prediction worker thread started.")

        # Raise thread priority on Windows if possible
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), 2   # ABOVE_NORMAL
            )
        except Exception:
            pass

        while self._running:
            try:
                seq = self._frame_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._store.is_loaded(self._lang):
                self._frame_q.task_done() if False else None
                continue

            word, conf = self._store.predict(self._lang, seq)
            sm_word, sm_conf = self._smoother.update(word, conf)

            if sm_word and sm_conf >= INFERENCE_THRESHOLD:
                pred = Prediction(
                    word=sm_word, confidence=sm_conf, lang=self._lang
                )
                self._last_pred = pred

                # Avoid repeating the same word consecutively
                if not self._text_buf.current_partial.endswith(sm_word):
                    self._text_buf.push_word(sm_word)

                result = InferenceResult(
                    prediction=pred,
                    buffered_text=self._text_buf.current_partial,
                    is_sentence_complete=False,
                )
                try:
                    self._result_q.put_nowait(result)
                except queue.Full:
                    try:
                        self._result_q.get_nowait()
                        self._result_q.put_nowait(result)
                    except Exception:
                        pass

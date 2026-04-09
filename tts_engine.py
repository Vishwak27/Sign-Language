"""
tts_engine.py — Non-Blocking Text-to-Speech Engine
====================================================
Runs TTS in a Daemon thread backed by a priority queue.
Supports offline (pyttsx3) and online (gTTS + pygame) backends.

Usage:
    engine = TTSEngine(backend="pyttsx3")
    engine.start()
    engine.speak("Hello world", emotion=EmotionState(label="happy"))
    engine.stop()
"""

import queue
import logging
import threading
import time
import os
import io
import tempfile
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass(order=True)
class SpeechTask:
    """Queue item for the TTS daemon thread."""
    priority:   int    = field(default=1, compare=True)   # Lower = higher priority
    text:       str    = field(default="", compare=False)
    lang:       str    = field(default="en", compare=False)
    rate:       int    = field(default=155,  compare=False)
    volume:     float  = field(default=0.85, compare=False)


# ─────────────────────────────────────────────────────────────
# PYTTSX3 BACKEND
# ─────────────────────────────────────────────────────────────

class _Pyttsx3Backend:
    def __init__(self):
        self._engine = None
        self._init()

    def _init(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   155)
            self._engine.setProperty("volume", 0.85)
            log.info("pyttsx3 TTS backend initialized.")
        except Exception as e:
            log.error(f"pyttsx3 init failed: {e}")
            self._engine = None

    def speak(self, task: SpeechTask):
        if self._engine is None:
            return
        try:
            self._engine.setProperty("rate",   task.rate)
            self._engine.setProperty("volume", task.volume)
            self._engine.say(task.text)
            self._engine.runAndWait()
        except Exception as e:
            log.error(f"pyttsx3 speak error: {e}")
            # Attempt re-init on failure
            self._init()

    def stop(self):
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
# GTTS BACKEND (Async, requires pygame for playback)
# ─────────────────────────────────────────────────────────────

class _GttsBackend:
    def __init__(self):
        self._pygame_ok = False
        try:
            import pygame
            pygame.mixer.init()
            self._pygame_ok = True
            log.info("gTTS + pygame backend initialized.")
        except Exception as e:
            log.warning(f"pygame init failed: {e}. gTTS output will be silent.")

    def speak(self, task: SpeechTask):
        if not self._pygame_ok:
            return
        try:
            from gtts import gTTS
            import pygame

            tts_obj = gTTS(text=task.text, lang=task.lang, slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tts_obj.save(tmp.name)
                tmp_path = tmp.name

            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.set_volume(task.volume)
            pygame.mixer.music.play()

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)

            os.unlink(tmp_path)
        except Exception as e:
            log.error(f"gTTS speak error: {e}")

    def stop(self):
        try:
            import pygame
            pygame.mixer.music.stop()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────
# TTS ENGINE — PUBLIC API
# ─────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Non-blocking TTS engine backed by a Daemon thread.

    The daemon thread consumes SpeechTask objects from a priority queue.
    The main (camera/UI) thread is never blocked.

    Args:
        backend:   "pyttsx3" (offline) or "gtts" (online)
        max_queue: Maximum pending speech tasks before dropping oldest.
    """

    def __init__(self, backend: str = "pyttsx3", max_queue: int = 5):
        self._backend_name = backend
        self._max_queue    = max_queue
        self._queue        = queue.PriorityQueue(maxsize=max_queue)
        self._muted        = False
        self._running      = False
        self._thread: Optional[threading.Thread] = None
        self._backend      = None

    # ── Lifecycle ────────────────────────────────────────────

    def start(self):
        """Initialize backend and start the daemon thread."""
        if self._backend_name == "gtts":
            self._backend = _GttsBackend()
        else:
            self._backend = _Pyttsx3Backend()

        self._running = True
        self._thread  = threading.Thread(
            target=self._worker,
            daemon=True,
            name="SpeechDaemon",
        )
        self._thread.start()
        log.info(f"TTSEngine started (backend={self._backend_name}).")

    def stop(self):
        """Signal the daemon thread to stop."""
        self._running = False
        if self._backend:
            self._backend.stop()
        log.info("TTSEngine stopped.")

    # ── Public Methods ────────────────────────────────────────

    def speak(
        self,
        text:     str,
        lang:     str   = "en",
        rate:     int   = 155,
        volume:   float = 0.85,
        priority: int   = 1,
    ):
        """
        Enqueue a speech task.  Returns immediately (non-blocking).
        Drops the task silently if muted or queue is full.
        """
        if self._muted or not text.strip():
            return

        task = SpeechTask(
            priority=priority,
            text=text,
            lang=lang,
            rate=rate,
            volume=volume,
        )
        try:
            self._queue.put_nowait(task)
        except queue.Full:
            log.debug("Speech queue full — dropping oldest entry.")
            try:
                self._queue.get_nowait()   # Drop oldest
                self._queue.put_nowait(task)
            except Exception:
                pass

    def speak_with_emotion(self, text: str, emotion_state, lang: str = "en"):
        """Speak text with TTS parameters derived from the emotion state."""
        self.speak(
            text,
            lang=lang,
            rate=emotion_state.tts_rate,
            volume=emotion_state.tts_volume,
        )

    def toggle_mute(self) -> bool:
        """Toggle mute state. Returns new muted status."""
        self._muted = not self._muted
        if self._muted and self._backend:
            self._backend.stop()
        log.info(f"TTS {'muted' if self._muted else 'unmuted'}.")
        return self._muted

    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    # ── Daemon Worker ─────────────────────────────────────────

    def _worker(self):
        """Daemon thread: continuously drain the speech queue."""
        while self._running:
            try:
                task = self._queue.get(timeout=0.2)
                if self._backend and not self._muted:
                    self._backend.speak(task)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as exc:
                log.error(f"TTS worker error: {exc}")

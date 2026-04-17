"""
extract_landmarks.py — Multi-threaded Landmark Extraction Pipeline
==================================================================
Converts raw video files from ASL / ISL / CSL datasets into
.npy landmark arrays ready for model training.

Usage:
    python extract_landmarks.py --lang all        # Process all languages
    python extract_landmarks.py --lang ASL        # Process ASL only
    python extract_landmarks.py --lang ISL --workers 8
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import argparse
import logging
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import time

# Local
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    ISL_DATASET_DIR,
    LANDMARK_DIR, MP_CONFIG, SEQUENCE_LENGTH,
    POSE_FEATURES, FACE_FEATURES, LH_FEATURES, RH_FEATURES,
    TOTAL_FEATURES, LOGS_DIR
)

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "extract_landmarks.log")),
    ]
)
log = logging.getLogger(__name__)

_progress_lock = Lock()
_stats = {"processed": 0, "failed": 0, "skipped": 0}


# ─────────────────────────────────────────────────────────────
# MEDIAPIPE HELPER
# ─────────────────────────────────────────────────────────────
def build_holistic():
    """Create a fresh MediaPipe Holistic instance (one per thread)."""
    mp_holistic = mp.solutions.holistic
    return mp_holistic.Holistic(**MP_CONFIG)


def extract_and_normalize_isl_keypoints(frames_results) -> np.ndarray:
    """
    Extracts hand landmarks specifically optimized for Indian Sign Language (ISL),
    flattens them, and normalizes them based on wrist position for distance-invariance.
    """
    # 2 hands * 21 landmarks * 3 coords = 126 features
    landmarks_data = np.zeros(126) 
    
    if frames_results.left_hand_landmarks:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in frames_results.left_hand_landmarks.landmark])
        wrist = coords[0]
        norm = coords - wrist
        max_val = np.max(np.abs(norm))
        if max_val > 0: norm = norm / max_val
        landmarks_data[0:63] = norm.flatten()
        
    if frames_results.right_hand_landmarks:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in frames_results.right_hand_landmarks.landmark])
        wrist = coords[0]
        norm = coords - wrist
        max_val = np.max(np.abs(norm))
        if max_val > 0: norm = norm / max_val
        landmarks_data[63:126] = norm.flatten()

    return landmarks_data

def extract_keypoints(results) -> np.ndarray:
    """
    Flatten all MediaPipe landmark arrays into a single 1-D NumPy vector.
    Shape: (TOTAL_FEATURES,)  i.e. (1629,)
    """
    # Pose — 33 × 3 = 99
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(POSE_FEATURES)

    # Face — 468 × 3 = 1404
    if results.face_landmarks:
        face = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(FACE_FEATURES)

    # Left hand — 21 × 3 = 63
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(LH_FEATURES)

    # Right hand — 21 × 3 = 63
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(RH_FEATURES)

    return np.concatenate([pose, face, lh, rh])  # (1629,)


def temporal_smooth(sequence: list, window: int = 3) -> list:
    """
    Apply a moving-average temporal smoothing to each frame in the sequence
    to reduce flickering in gesture detection.
    """
    if len(sequence) < window:
        return sequence
    smoothed = []
    arr = np.array(sequence)       # (T, features)
    for i in range(len(arr)):
        start = max(0, i - window // 2)
        end   = min(len(arr), i + window // 2 + 1)
        smoothed.append(arr[start:end].mean(axis=0))
    return smoothed


# ─────────────────────────────────────────────────────────────
# CORE EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_from_video(video_path: str, label: str, out_dir: str,
                       sequence_len: int = SEQUENCE_LENGTH) -> bool:
    """
    Extract a fixed-length landmark sequence from a single video file.
    Saves the result as a .npy file.

    Returns True on success, False on failure.
    """
    out_dir_path = Path(out_dir) / label
    out_dir_path.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem
    out_file   = out_dir_path / f"{video_name}.npy"

    # Skip already-processed files
    if out_file.exists():
        with _progress_lock:
            _stats["skipped"] += 1
        return True

    holistic = build_holistic()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        with _progress_lock:
            _stats["failed"] += 1
        holistic.close()
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, max(total_frames - 1, 0),
                                sequence_len, dtype=int)
    sequence = []

    try:
        frame_idx = 0
        target_set = set(frame_indices.tolist())

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in target_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)
                kp = extract_keypoints(results)
                sequence.append(kp)

            frame_idx += 1
            if len(sequence) >= sequence_len:
                break

    finally:
        cap.release()
        holistic.close()

    # Pad if video was shorter than sequence_len
    while len(sequence) < sequence_len:
        pad = sequence[-1] if sequence else np.zeros(TOTAL_FEATURES)
        sequence.append(pad)

    # Temporal smoothing
    sequence = temporal_smooth(sequence)

    np_seq = np.array(sequence[:sequence_len])  # (30, 1629)
    np.save(str(out_file), np_seq)

    with _progress_lock:
        _stats["processed"] += 1

    return True


# ─────────────────────────────────────────────────────────────
# DATASET SCANNER
# ─────────────────────────────────────────────────────────────
def scan_dataset(dataset_dir: str, lang: str) -> list[dict]:
    """
    Walk dataset_dir and collect (video_path, label) pairs.
    Expected structure:
        dataset_dir/
          <class_label>/
            video1.mp4
            video2.avi
            ...
    """
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    tasks = []
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        log.warning(f"[{lang}] Dataset directory not found: {dataset_dir}")
        log.warning(f"[{lang}] Place your videos in: {dataset_dir}/<class_label>/video.mp4")
        return tasks

    for class_dir in dataset_path.iterdir():
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for video_file in class_dir.rglob("*"):
            if video_file.suffix.lower() in video_extensions:
                tasks.append({
                    "video_path": str(video_file),
                    "label":      label,
                    "lang":       lang,
                })

    log.info(f"[{lang}] Found {len(tasks)} videos across {len(set(t['label'] for t in tasks))} classes.")
    return tasks


# ─────────────────────────────────────────────────────────────
# PARALLEL PROCESSING
# ─────────────────────────────────────────────────────────────
def process_language(lang: str, dataset_dir: str, workers: int = 4):
    """Process all videos for a single sign language."""
    out_dir = os.path.join(LANDMARK_DIR, lang)
    os.makedirs(out_dir, exist_ok=True)

    tasks = scan_dataset(dataset_dir, lang)
    if not tasks:
        log.warning(f"[{lang}] No videos found — skipping.")
        return

    log.info(f"[{lang}] Starting extraction with {workers} workers …")

    def _worker(task):
        return extract_from_video(
            task["video_path"],
            task["label"],
            out_dir,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        with tqdm(total=len(tasks), desc=f"[{lang}]", unit="vid") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    log.error(f"Worker error: {exc}")
                    with _progress_lock:
                        _stats["failed"] += 1
                pbar.update(1)

    # Save class manifest
    classes = sorted(set(t["label"] for t in tasks))
    manifest = {
        "language":       lang,
        "classes":        classes,
        "num_classes":    len(classes),
        "total_samples":  len(tasks),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim":    TOTAL_FEATURES,
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"[{lang}] Manifest saved -> {manifest_path}")


# ─────────────────────────────────────────────────────────────
# GENERATE DEMO DATA (when no real datasets present)
# ─────────────────────────────────────────────────────────────
def generate_demo_landmarks(lang: str, classes: list, samples_per_class: int = 50):
    """
    Generate synthetic landmark data for testing when real datasets
    are not yet available. Creates realistic-looking random sequences.
    """
    out_dir = Path(LANDMARK_DIR) / lang
    log.info(f"[{lang}] Generating {samples_per_class} synthetic samples for {len(classes)} classes …")

    for cls in tqdm(classes, desc=f"[{lang}] Generating", unit="class"):
        cls_dir = out_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_class):
            # Simulate a gesture: base pose + class-specific oscillation
            base = np.random.randn(TOTAL_FEATURES) * 0.1
            seq  = []
            for t in range(SEQUENCE_LENGTH):
                noise     = np.random.randn(TOTAL_FEATURES) * 0.02
                variation = np.sin(2 * np.pi * t / SEQUENCE_LENGTH + hash(cls) % 100) * 0.05
                seq.append(base + noise + variation)

            np_seq = np.array(seq)
            np.save(str(cls_dir / f"sample_{i:04d}.npy"), np_seq)

    # Save manifest
    manifest = {
        "language":       lang,
        "classes":        classes,
        "num_classes":    len(classes),
        "total_samples":  len(classes) * samples_per_class,
        "sequence_length": SEQUENCE_LENGTH,
        "feature_dim":    TOTAL_FEATURES,
        "is_synthetic":   True,
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path = out_dir / "manifest.json"
    with open(str(manifest_path), "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"[{lang}] Synthetic data ready -> {out_dir}")


# ─────────────────────────────────────────────────────────────
# DEMO CLASS VOCABULARIES
# ─────────────────────────────────────────────────────────────
ISL_DEMO_CLASSES = [
    "namaste", "shukriya", "haan", "nahi", "roko", "pani", "khana",
    "madad", "achha", "bura", "main", "tum", "pyaar", "kahan", "kya",
    "school", "ghar", "kaam", "aao", "jao",
]

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal SLR — Landmark Extraction Pipeline"
    )
    parser.add_argument("--lang",    default="ISL",
                        choices=["ISL"],
                        help="Which language dataset to process")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker threads")
    parser.add_argument("--demo",    action="store_true",
                        help="Generate synthetic demo data (no real dataset needed)")
    args = parser.parse_args()

    lang_map = {
        "ISL": (ISL_DATASET_DIR, ISL_DEMO_CLASSES),
    }

    langs = list(lang_map.keys()) if args.lang == "all" else [args.lang]

    for lang in langs:
        dataset_dir, demo_classes = lang_map[lang]
        tasks = scan_dataset(dataset_dir, lang)

        if not tasks or args.demo:
            log.info(f"[{lang}] No real dataset found or --demo flag set. "
                     f"Generating synthetic landmarks …")
            generate_demo_landmarks(lang, demo_classes)
        else:
            process_language(lang, dataset_dir, workers=args.workers)

    log.info("=" * 60)
    log.info(f"EXTRACTION COMPLETE")
    log.info(f"  Processed : {_stats['processed']}")
    log.info(f"  Skipped   : {_stats['skipped']}  (already existed)")
    log.info(f"  Failed    : {_stats['failed']}")
    log.info(f"  Output    : {LANDMARK_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

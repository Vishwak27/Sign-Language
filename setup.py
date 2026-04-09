"""
setup.py — One-Shot Environment Verification & Setup
=====================================================
Run this script first to verify all dependencies are installed.
It will also create the required directory structure and
generate a synthetic demo dataset for immediate testing.

Usage:
    python setup.py
    python setup.py --skip-demo   # Skip synthetic data generation
"""

import subprocess
import sys
import os
import importlib
import argparse
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# ANSI Colors
# ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✓ {msg}{RESET}")
def fail(msg):  print(f"  {RED}✗ {msg}{RESET}")
def warn(msg):  print(f"  {YELLOW}⚠ {msg}{RESET}")
def info(msg):  print(f"  {CYAN}→ {msg}{RESET}")
def header(msg): print(f"\n{BOLD}{'─'*55}\n  {msg}\n{'─'*55}{RESET}")


# ─────────────────────────────────────────────────────────────
# DEPENDENCY CHECKS
# ─────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    ("cv2",          "opencv-python"),
    ("mediapipe",    "mediapipe"),
    ("tensorflow",   "tensorflow"),
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("sklearn",      "scikit-learn"),
    ("scipy",        "scipy"),
    ("pyttsx3",      "pyttsx3"),
    ("gtts",         "gTTS"),
    ("pygame",       "pygame"),
    ("matplotlib",   "matplotlib"),
    ("seaborn",      "seaborn"),
    ("PIL",          "Pillow"),
    ("tqdm",         "tqdm"),
]

def check_dependencies():
    header("Checking Python Dependencies")
    missing = []
    for module, package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module)
            ok(f"{package}")
        except ImportError:
            fail(f"{package}  ← NOT INSTALLED")
            missing.append(package)
    return missing


def install_missing(missing: list):
    if not missing:
        return
    header("Installing Missing Packages")
    for pkg in missing:
        info(f"Installing {pkg} …")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            ok(f"{pkg} installed.")
        else:
            fail(f"Failed to install {pkg}: {result.stderr[:200]}")


# ─────────────────────────────────────────────────────────────
# DIRECTORY STRUCTURE
# ─────────────────────────────────────────────────────────────

DIRECTORIES = [
    "datasets/WLASL",
    "datasets/Include",
    "datasets/DEVISIGN",
    "landmarks/ASL",
    "landmarks/ISL",
    "landmarks/CSL",
    "models",
    "reports",
    "logs",
]

def create_directories():
    header("Creating Directory Structure")
    base = Path(__file__).parent
    for d in DIRECTORIES:
        path = base / d
        path.mkdir(parents=True, exist_ok=True)
        ok(str(path.relative_to(base)))


# ─────────────────────────────────────────────────────────────
# VERSION INFO
# ─────────────────────────────────────────────────────────────

def print_versions():
    header("Installed Versions")
    try:
        import cv2;         ok(f"OpenCV      {cv2.__version__}")
    except Exception:       warn("OpenCV not available")
    try:
        import mediapipe as mp; ok(f"MediaPipe   {mp.__version__}")
    except Exception:       warn("MediaPipe not available")
    try:
        import tensorflow as tf: ok(f"TensorFlow  {tf.__version__}")
    except Exception:       warn("TensorFlow not available")
    try:
        import numpy as np; ok(f"NumPy       {np.__version__}")
    except Exception:       warn("NumPy not available")
    try:
        import sklearn;     ok(f"Scikit-Learn {sklearn.__version__}")
    except Exception:       warn("Scikit-learn not available")
    try:
        import pyttsx3;     ok(f"pyttsx3     {pyttsx3.__version__}")
    except Exception:       warn("pyttsx3 not available")

    print()
    ok(f"Python {sys.version.split()[0]}")


# ─────────────────────────────────────────────────────────────
# WEBCAM CHECK
# ─────────────────────────────────────────────────────────────

def check_webcam():
    header("Webcam Detection")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ok("Webcam found at index 0.")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            f = cap.get(cv2.CAP_PROP_FPS)
            info(f"Resolution: {w}×{h} @ {f:.0f} FPS")
            cap.release()
        else:
            warn("No webcam found at index 0. The app will show an error when launched.")
    except Exception as e:
        warn(f"Could not check webcam: {e}")


# ─────────────────────────────────────────────────────────────
# GENERATE DEMO DATA
# ─────────────────────────────────────────────────────────────

def generate_demo_data():
    header("Generating Synthetic Demo Dataset")
    info("This creates fake landmark data so you can test the full pipeline")
    info("without needing the real WLASL / Include / DEVISIGN datasets.")
    print()
    result = subprocess.run(
        [sys.executable, "extract_landmarks.py", "--demo", "--lang", "all"],
        text=True
    )
    if result.returncode == 0:
        ok("Demo data generated successfully.")
    else:
        fail("Demo data generation failed. See output above.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SLR System Setup")
    parser.add_argument("--skip-demo",    action="store_true",
                        help="Skip synthetic data generation.")
    parser.add_argument("--skip-install", action="store_true",
                        help="Skip auto-installing missing packages.")
    args = parser.parse_args()

    print(f"\n{BOLD}{'='*55}")
    print("  SLR Studio — Environment Setup & Verification")
    print(f"{'='*55}{RESET}")

    # 1. Check dependencies
    missing = check_dependencies()

    # 2. Install missing (unless skipped)
    if missing and not args.skip_install:
        install_missing(missing)
        missing = check_dependencies()   # Re-check after install

    # 3. Create directories
    create_directories()

    # 4. Print versions
    print_versions()

    # 5. Webcam
    check_webcam()

    # 6. Generate demo data
    if not args.skip_demo:
        generate_demo_data()

    # Final summary
    print(f"\n{BOLD}{'='*55}")
    if not missing:
        print(f"{GREEN}  ✓ ALL SYSTEMS READY{RESET}")
        print(f"\n{CYAN}  Next steps:{RESET}")
        print(f"    1. python train_slr_models.py     ← Train models")
        print(f"    2. python app.py                  ← Launch real-time SLR")
        print()
        print(f"  {YELLOW}To use real datasets:{RESET}")
        print(f"    Place videos in:  datasets/WLASL/<class>/video.mp4")
        print(f"    Then run:         python extract_landmarks.py --lang ASL")
    else:
        print(f"{RED}  ✗ Some packages failed to install:{RESET}")
        for p in missing:
            print(f"    pip install {p}")
    print(f"{BOLD}{'='*55}{RESET}\n")


if __name__ == "__main__":
    main()

# SLR Studio — Multi-Modal Sign Language Recognition

> **Real-time ASL · ISL · CSL recognition with Sign-to-Text & Sign-to-Speech output**
> Built with the Antigravity Framework — 2026

---

## ⚡ Quick Start (3 commands)

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Set up environment + generate synthetic demo data
python setup.py

# 3. Train demo models (uses synthetic data if no real dataset)
python train_slr_models.py

# 4. Launch the real-time app
python app.py
```

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│           SLR Studio — Antigravity Framework               │
├────────────────────────────────────────────────────────────┤
│ Thread 1 — MainThread (Camera + OpenCV UI)                 │
│   • Captures frames at 30 FPS                              │
│   • Runs MediaPipe Holistic for landmark extraction        │
│   • Renders skeleton overlay + full HUD                    │
├────────────────────────────────────────────────────────────┤
│ Thread 2 — PredictionAsyncWorker (High Priority)           │
│   • Bi-LSTM inference on 30-frame landmark sequences       │
│   • Temporal smoothing + prediction smoothing              │
│   • Buffer & Flush for complete sentence detection         │
├────────────────────────────────────────────────────────────┤
│ Thread 3 — SpeechDaemon                                    │
│   • pyttsx3 (offline) or gTTS (online) TTS                │
│   • Priority queue — NEVER blocks the UI thread           │
│   • Emotion-modulated pitch/rate/volume                    │
└────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Sign Language Project/
├── app.py                  ← Main real-time application
├── config.py               ← All settings (landmarks, model, UI)
├── extract_landmarks.py    ← Multi-threaded video → .npy pipeline
├── train_slr_models.py     ← Three-headed Bi-LSTM training
├── inference_engine.py     ← Async inference + text buffer
├── emotion_detector.py     ← Facial landmark → emotion state
├── tts_engine.py           ← Non-blocking TTS daemon
├── setup.py                ← Environment verification
├── requirements.txt        ← All dependencies
│
├── datasets/
│   ├── WLASL/              ← ASL dataset (place videos here)
│   │   └── <class>/video.mp4
│   ├── Include/            ← ISL dataset
│   └── DEVISIGN/           ← CSL dataset
│
├── landmarks/              ← Extracted .npy landmark files
│   ├── ASL/
│   ├── ISL/
│   └── CSL/
│
├── models/                 ← Trained model files
│   ├── base_asl_model.h5
│   ├── isl_model.h5
│   ├── csl_model.h5
│   └── *.tflite
│
├── reports/                ← model_performance.md + charts
└── logs/                   ← Runtime logs
```

---

## 🎯 Training Pipeline

### Step 1 – Extract Landmarks

```bash
# Using real datasets
python extract_landmarks.py --lang ASL --workers 8
python extract_landmarks.py --lang ISL --workers 8
python extract_landmarks.py --lang CSL --workers 8

# OR generate synthetic demo data (no real dataset needed)
python extract_landmarks.py --demo --lang all
```

### Step 2 – Train the Three-Headed Model

```bash
# Train all three (ASL base → ISL fine-tune → CSL fine-tune)
python train_slr_models.py

# Train only ASL base
python train_slr_models.py --lang ASL

# Skip base training, fine-tune with existing checkpoint
python train_slr_models.py --skip-base
```

### Step 3 – View Results

Reports are saved to `reports/model_performance.md`

---

## 🖥️ Real-Time App Controls

| Key | Action |
|-----|--------|
| `1` | Cycle language: ASL → ISL → CSL |
| `S` | Toggle speech mute/unmute |
| `R` | Reset text buffer |
| `Q` / `ESC` | Quit |

---

## 🧠 Model Architecture

```
Input (30 frames × 1629 features)
    │
    ├─ Masking Layer (handles zero-padded frames)
    │
    ├─ BiLSTM(128) + BatchNorm
    │
    ├─ BiLSTM(64)  + BatchNorm + Dropout(0.4)
    │
    ├─ Dense(64) → Dropout
    │
    ├─ Dense(32) → Dropout
    │
    └─ Softmax(num_classes)
```

**Transfer Learning Strategy:**
1. Train **ASL Base Model** on the full WLASL dataset (largest)
2. Freeze BiLSTM backbone, replace + fine-tune **ISL Head** (two-handed, SOV grammar)
3. Freeze BiLSTM backbone, replace + fine-tune **CSL Head** (stroke-based gestures)

---

## 🎙️ TTS & Emotion

The emotion detector analyzes MediaPipe face mesh landmarks in real-time:

| Emotion | Trigger | TTS Rate | Volume | Pitch |
|---------|---------|----------|--------|-------|
| Neutral | Default | 155 wpm | 85% | 1.0× |
| Happy | Wide mouth + raised brows | 170 wpm | 90% | 1.1× |
| Angry | Narrowed eyes + open mouth | 190 wpm | 100% | 0.85× |
| Sad | Drooped eyes + closed mouth | 130 wpm | 70% | 0.95× |
| Questioning | Raised brows + slight open | 140 wpm | 75% | 1.15× |

---

## 📦 Datasets (Real Data)

| Language | Dataset | Source |
|----------|---------|--------|
| ASL | WLASL-2000 | https://dxli94.github.io/WLASL/ |
| ISL | INCLUDE-50 | https://zenodo.org/record/4010759 |
| CSL | DEVISIGN-G | http://vipl.ict.ac.cn/resources/databases/ |

Place videos in `datasets/<LANG>/<class_label>/video.mp4`

---

## ⚙️ Configuration

All settings are in `config.py`:

- `SEQUENCE_LENGTH` — frames per gesture (default: 30)
- `INFERENCE_THRESHOLD` — min confidence to accept (default: 0.75)
- `PAUSE_TIMEOUT_SECONDS` — silence time before flushing sentence (default: 1.8s)
- `MP_CONFIG` — MediaPipe Holistic settings
- `EMOTION_TTS_PARAMS` — TTS parameter tuning per emotion

---

## 🔧 Requirements

- Python 3.10+
- Webcam
- 8GB RAM minimum (16GB recommended for training)
- GPU optional but recommended for training (CUDA 12.x)

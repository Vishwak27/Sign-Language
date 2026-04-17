# ISL SignSense — Indian Sign Language Learning Platform

> **Real-time ISL recognition with an interactive Gamified Quiz Mode**
> Built with MediaPipe, FastAPI, and React — 2026

Welcome to the beginner-centric ISL learning platform. Designing for Indian Sign Language (ISL) requires a dedicated approach because ISL relies heavily on two-handed, spatially complex signs (unlike ASL). This platform employs specific distance-invariant wrist-normalization techniques to accurately track these signs.

---

## ⚡ Quick Start

### 1. Backend API (FastAPI + WebSockets)
The backend does the heavy lifting for real-time video processing and feature extraction.

```bash
# 1. Install all dependencies
pip install -r requirements.txt fastapi uvicorn websockets opencv-python mediapipe

# 2. Set up environment + generate synthetic demo data
python setup.py

# 3. Launch the Real-Time API Bridge
uvicorn fastapi_server:app --reload --port 8000
```

### 2. Frontend Interface (React/Next.js)
The frontend serves the gamified "Quiz Mode" UI to users.

```bash
cd frontend
npm install
npm install lucide-react react-webcam
npm run dev
```

---

## 🏗️ Architecture

```text
┌────────────────────────────────────────────────────────────┐
│                    ISL SignSense Platform                  │
├────────────────────────────────────────────────────────────┤
│ Client-Side (React)                                        │
│   • Captures webcam frames at ~15-30 FPS                   │
│   • Renders Gamified Dashboard & Live Confidence HUD       │
│   • Streams frames to backend via WebSockets               │
├────────────────────────────────────────────────────────────┤
│ Server-Side (FastAPI WebSocket Bridge)                     │
│   • Parses Base64 frames using OpenCV                      │
│   • Runs MediaPipe Holistic Hand Tracking                  │
│   • Normalizes 2-handed coordinates relative to the wrist  │
│   • Scores the gesture and streams confidence back to UI   │
└────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```text
Sign Language Project/
├── frontend/src/App.jsx    ← React Gamified Quiz Dashboard
├── fastapi_server.py       ← Real-time inference WebSocket API
├── config.py               ← Configurations for Data & Training
├── extract_landmarks.py    ← ISL Wrist-Normalized .npy generator
├── train_slr_models.py     ← Core BiLSTM Training script
├── inference_engine.py     ← Continuous Inference Engine
├── app.py                  ← Legacy Streamlit Local App
├── setup.py                ← Environment verification & Mock data
└── requirements.txt        
```

---

## 🎯 Model Training Pipeline (ISL Focus)

**Step 1 – Extract Landmarks (With ISL Normalization)**
```bash
# Using real dataset (Place videos in datasets/Include/<class>/video.mp4)
python extract_landmarks.py

# OR generate synthetic demo data (no real dataset needed)
python extract_landmarks.py --demo
```

**Step 2 – Train the ISL Model**
```bash
python train_slr_models.py
```

Reports are saved to `reports/model_performance.md`

---

## ☁️ AWS Deployment Strategy

To deploy this platform with minimal "video lag", serverless cold starts must be avoided.

**1. Frontend (React UI)**
*   Host the static interactive app on **AWS Amplify**.
*   This uses Amazon CloudFront globally to serve assets instantly to users.

**2. Backend API (Persistent WebSocket Inference)**
*   **Do not use AWS Lambda** for the WebSocket inference loop, as cold-starts heavily degrade the real-time feedback ring.
*   Package the FastAPI app in a Docker container and deploy it to **Amazon ECS with AWS Fargate**. This keeps the WebSocket server continuously running and easily auto-scalable based on concurrent learners.

**3. Datasets & Models**
*   **Amazon S3** for persistent storage of new training videos and compiled `.keras`/`.tflite` model files.

---

## 📦 Datasets (Real Data)

| Language | Dataset | Source |
|----------|---------|--------|
| ISL | INCLUDE-50 | [Zenodo INCLUDE](https://zenodo.org/record/4010759) |

Place videos in `datasets/Include/<class_label>/video.mp4`

---

## ⚙️ Configuration

All ML logic settings are in `config.py`:
- `SEQUENCE_LENGTH` — frames per gesture (default: 30)
- `INFERENCE_THRESHOLD` — min confidence to accept (default: 0.75)

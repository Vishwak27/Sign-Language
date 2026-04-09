"""
app.py — Antigravity Streamlit Frontend
========================================
Production-ready responsive web UI for SLR Studio.
Uses Native Streamlit Auth (st.login).
"""

import os
import av
import cv2
import queue
import logging
import threading
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# Model Imports
import mediapipe as mp
from config import FACE_LANDMARKS, POSE_FEATURES, FACE_FEATURES, LH_FEATURES, RH_FEATURES
from inference_engine import InferenceEngine
from tts_engine import TTSEngine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── 1. PAGE & STATE Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="SLR Studio | NeuralACT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = False
if "current_lang" not in st.session_state:
    st.session_state.current_lang = "ASL"

# ── 2. GLOBAL ENGINES ────────────────────────────────────────────────
@st.cache_resource
def get_inference_engine():
    engine = InferenceEngine(lang="ASL")
    engine.start()
    return engine

@st.cache_resource
def get_tts_engine():
    engine = TTSEngine(backend="pyttsx3")
    engine.start()
    return engine

engine = get_inference_engine()
tts    = get_tts_engine()

# MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_keypoints(results) -> np.ndarray:
    pose = (np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks else np.zeros(POSE_FEATURES))
    
    face = (np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
            if results.face_landmarks else np.zeros(FACE_FEATURES))
    
    lh = (np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
          if results.left_hand_landmarks else np.zeros(LH_FEATURES))
          
    rh = (np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
          if results.right_hand_landmarks else np.zeros(RH_FEATURES))
          
    return np.concatenate([pose, face, lh, rh])

# ── 3. AUTHENTICATION MODULE ──────────────────────────────────────────
# Streamlit experimental or native login
def render_splash():
    st.title("NeuralACT Sign Language Studio")
    st.markdown("### Welcome to the world's most advanced Multi-lingual SLR Platform.")
    st.image("https://images.unsplash.com/photo-1543886532-6020c2ceecb6?q=80&w=2670&auto=format&fit=crop", 
             use_container_width=True)
    st.info("You must be authenticated to access the Vision Hub.")
    if hasattr(st, "login"):
        st.login(provider="google")
    else:
        st.warning("st.login not available in this version. Proceeding in Dev Mode.")
        if st.button("Enter Developer Mode"):
            st.session_state.dev_mode = True
            st.rerun()

is_authed = False
user_name = "Developer"
user_icon = "🎓"

if hasattr(st, "experimental_user") and getattr(st.experimental_user, "is_logged_in", False):
    is_authed = True
    user_name = st.experimental_user.name
# Fallback for dev mode
elif st.session_state.get("dev_mode", False):
    is_authed = True

if not is_authed:
    render_splash()
    st.stop()

# ── 4. MAIN USER INTERFACE ────────────────────────────────────────────

with st.sidebar:
    st.header(f"{user_icon} Welcome, {user_name}!")
    if hasattr(st, "logout"):
        st.logout()
    
    st.divider()
    st.subheader("⚙️ Settings")
    
    new_lang = st.selectbox("Language Engine", ["ASL", "ISL", "CSL"], index=["ASL", "ISL", "CSL"].index(st.session_state.current_lang))
    if new_lang != st.session_state.current_lang:
        st.session_state.current_lang = new_lang
        engine.set_language(new_lang)
    
    tts_toggle = st.toggle("Voice Synthesis Engine", value=st.session_state.tts_enabled)
    st.session_state.tts_enabled = tts_toggle
    
    with st.expander("📖 ISL Instructions"):
        st.markdown("""
        **Indian Sign Language (Two-Handed)**
        - Ensure both hands are visible in the frame.
        - Stand at least 2 feet away from the lens.
        - ISL relies heavily on Subject-Object-Verb (SOV) structure.
        - Ensure good lighting for facial expressions!
        """)

st.title("Vision Hub & Real-time Translation")
st.markdown("Stream your camera feed below. Processing happens entirely on the local device without latency overhead.")

# ── 5. WEBRTC VIDEO PROCESSOR ─────────────────────────────────────────

class SLRVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            refine_face_landmarks=False
        )
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Process via MediaPipe
        results = self.holistic.process(img_rgb)
        
        # 2. Extract and Push to Engine
        kp = extract_keypoints(results)
        engine.push_frame(kp)
        
        # 3. Pull Result
        pred_res = engine.get_result()
        
        # 4. Handle Text and Speech
        active_text = ""
        color = (200, 200, 200)
        
        if pred_res:
            active_text = pred_res.buffered_text
            
            if pred_res.is_sentence_complete and active_text:
                if st.session_state.tts_enabled:
                    tts.queue_speech(active_text, lang=st.session_state.current_lang)
                color = (0, 255, 0)
            elif pred_res.prediction and pred_res.prediction.confidence > 0.7:
                color = (255, 100, 100) # Active word typing
        
        # 5. Draw Skeleton
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
        # 6. Overlay HUD 
        h, w, _ = img.shape
        cv2.rectangle(img, (0, h-80), (w, h), (10, 17, 40), -1)
        
        # Display Text
        cv2.putText(img, active_text if active_text else "Sign something...", 
                    (20, h-30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


stream_ctx = webrtc_streamer(
    key="slr-streamer",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=SLRVideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.subheader("Console Output")
st.code("System is actively syncing parallel threads. Waiting for inference module...")

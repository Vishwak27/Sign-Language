import base64
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import random

app = FastAPI(title="ISL SignSense Bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def extract_and_normalize_landmarks(frame):
    """
    Extracts hand landmarks, flattens them, and normalizes them based on the 
    bounding box of the hands to make the model distance-invariant.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    
    # 2 hands * 21 landmarks * 3 coords = 126 features
    landmarks_data = np.zeros(126) 
    
    found_hands = False
    
    if results.multi_hand_landmarks and results.multi_handedness:
        found_hands = True
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label # 'Left' or 'Right'
            offset = 0 if hand_label == 'Left' else 63
            
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Normalization logic: shift relative to wrist (coord 0) and scale
            wrist = coords[0]
            normalized_coords = coords - wrist
            max_val = np.max(np.abs(normalized_coords))
            if max_val > 0:
                normalized_coords = normalized_coords / max_val
                
            landmarks_data[offset:offset+63] = normalized_coords.flatten()
            
    return landmarks_data, found_hands

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected via WebSocket")
    
    # Simple state mock for gamified experience
    current_confidence = 0
    try:
        while True:
            data_str = await websocket.receive_text()
            data = json.loads(data_str)
            
            if data["type"] == "FRAME":
                img_data = data["data"]
                target = data["target"]
                
                # Decode image
                encoded_data = img_data.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Preprocess specifically for ISL Two-Handed
                landmarks, found = extract_and_normalize_landmarks(img)
                
                # Send mock confidence logic for demonstration
                # In production: run model model.predict(landmarks)
                if found:
                    # Random logic to simulate confidence growing if hands are in frame
                    current_confidence = min(100, current_confidence + random.randint(5, 15))
                else:
                    current_confidence = max(0, current_confidence - random.randint(10, 20))
                    
                await websocket.send_json({
                    "type": "CONFIDENCE",
                    "score": current_confidence,
                    "target": target
                })
                
    except WebSocketDisconnect:
        print("Client disconnected")

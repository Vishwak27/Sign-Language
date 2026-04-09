import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Camera, Volume2, VolumeX, Radio, Sparkles, MessageSquareText } from 'lucide-react';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const [language, setLanguage] = useState('ASL');
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [activeText, setActiveText] = useState('');
  const [ws, setWs] = useState(null);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:5000');
    
    socket.onopen = () => {
      console.log('Connected to AI Bridge');
      setWs(socket);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'PREDICTION' && data.text) {
          setActiveText(data.text);
        }
      } catch (e) {
        console.error("Error parsing message", e);
      }
    };

    socket.onclose = () => console.log('Disconnected');
    
    return () => socket.close();
  }, []);

  const captureFrame = useCallback(() => {
    if (webcamRef.current && ws && ws.readyState === WebSocket.OPEN) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        ws.send(JSON.stringify({ type: 'FRAME', lang: language, data: imageSrc }));
      }
    }
  }, [webcamRef, ws, language]);

  useEffect(() => {
    const interval = setInterval(captureFrame, 100);
    return () => clearInterval(interval);
  }, [captureFrame]);

  return (
    <div className="dashboard">
      
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={24} color="#fff" />
          </div>
          <h1>SignSense AI</h1>
        </div>

        <div className="control-group">
          <span className="control-label">Sign Language Engine</span>
          <select 
            className="select-box"
            value={language} 
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="ASL">American Sign Language</option>
            <option value="ISL">Indian Sign Language</option>
            <option value="CSL">Chinese Sign Language</option>
          </select>
        </div>

        <div className="control-group">
          <span className="control-label">Accessibility</span>
          <button 
            className={`toggle-btn ${ttsEnabled ? 'active' : ''}`}
            onClick={() => setTtsEnabled(!ttsEnabled)}
          >
            {ttsEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
            Text-to-Speech Output
          </button>
        </div>

        <div className="status-indicator">
          <span className="dot"></span>
          Connected to local Engine
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-stage">
        
        {/* CAMERA HUB */}
        <div className="video-container">
          <Webcam
            ref={webcamRef}
            audio={false}
            screenshotFormat="image/jpeg"
            className="webcam"
            videoConstraints={{ facingMode: "user" }}
          />
          <div className="camera-overlay">
            <Radio size={16} color="#10b981" />
            Live Processing
          </div>
        </div>

        {/* NLP OUTPUT TERMINAL */}
        <div className="translation-box">
          <div className="trans-label">
            <MessageSquareText size={16} /> 
            Live Translation
          </div>
          <div className="trans-text">
            {activeText ? activeText : <span className="placeholder">Start signing into the camera...</span>}
          </div>
        </div>

      </main>
    </div>
  );
}

export default App;

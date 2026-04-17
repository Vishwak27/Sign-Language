import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Volume2, VolumeX, Sparkles, Trophy, CheckCircle, Target, ArrowRight, Play, LogOut } from 'lucide-react';
import { supabase } from './supabaseClient';
import './App.css';

const CURRICULUM = [
  { level: 1, title: 'The Basics', items: ['A', 'B', 'C', '1', '2'] },
  { level: 2, title: 'Greetings', items: ['Hello', 'Thank You', 'Please'] },
  { level: 3, title: 'Everyday', items: ['Water', 'Food', 'Stop'] }
];

function App() {
  const webcamRef = useRef(null);
  const [activeLevel, setActiveLevel] = useState(0);
  const [targetIdx, setTargetIdx] = useState(0);
  const [targetSign, setTargetSign] = useState(CURRICULUM[0].items[0]);
  const [confidence, setConfidence] = useState(0);
  const [streak, setStreak] = useState(0);
  const [score, setScore] = useState(0);
  const [isSuccess, setIsSuccess] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(false);
  const [ws, setWs] = useState(null);
  const [quizActive, setQuizActive] = useState(false);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:8000/ws');
    
    socket.onopen = () => {
      console.log('Connected to AI Bridge');
      setWs(socket);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'CONFIDENCE' && quizActive && !isSuccess) {
          setConfidence(data.score); // 0 to 100
          
          if (data.score >= 85) {
             handleSuccess();
          }
        }
      } catch (e) {
        console.error("Error parsing message", e);
      }
    };

    socket.onclose = () => console.log('Disconnected');
    
    return () => socket.close();
  }, [quizActive, isSuccess]);

  const captureFrame = useCallback(() => {
    if (webcamRef.current && ws && ws.readyState === WebSocket.OPEN && quizActive) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        ws.send(JSON.stringify({ 
          type: 'FRAME', 
          target: targetSign, 
          data: imageSrc 
        }));
      }
    }
  }, [webcamRef, ws, quizActive, targetSign]);

  useEffect(() => {
    const interval = setInterval(captureFrame, 150);
    return () => clearInterval(interval);
  }, [captureFrame]);

  const handleSuccess = () => {
    setIsSuccess(true);
    setScore(s => s + 10 + (streak * 2));
    setStreak(s => s + 1);
    
    // Play ding sound if possible (omitted for brevity)
    
    setTimeout(() => {
      nextSign();
      setIsSuccess(false);
      setConfidence(0);
    }, 2000);
  };

  const nextSign = () => {
    const currentList = CURRICULUM[activeLevel].items;
    if (targetIdx + 1 < currentList.length) {
      setTargetIdx(targetIdx + 1);
      setTargetSign(currentList[targetIdx + 1]);
    } else if (activeLevel + 1 < CURRICULUM.length) {
      setActiveLevel(activeLevel + 1);
      setTargetIdx(0);
      setTargetSign(CURRICULUM[activeLevel + 1].items[0]);
    } else {
      // Finished
      setQuizActive(false);
      setTargetSign("Course Complete!");
    }
  };

  return (
    <div className="dashboard">
      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Sparkles size={24} color="#fff" />
          </div>
          <h1>ISL Learn</h1>
        </div>

        <div className="stats-card">
           <div className="stat-item">
             <Trophy size={18} color="#facc15" />
             <span>Score: <strong>{score}</strong></span>
           </div>
           <div className="stat-item">
             <Target size={18} color="#ef4444" />
             <span>Streak: <strong>{streak}x</strong></span>
           </div>
        </div>

        <div className="curriculum">
          <h3 className="section-title">Curriculum</h3>
           {CURRICULUM.map((mod, idx) => (
             <div 
               key={idx} 
               className={`module-card ${idx === activeLevel ? 'active' : ''} ${idx < activeLevel ? 'completed' : ''}`}
               onClick={() => {
                 setActiveLevel(idx);
                 setTargetIdx(0);
                 setTargetSign(CURRICULUM[idx].items[0]);
                 setQuizActive(false);
               }}
             >
               <div className="module-header">
                 <span>Level {mod.level}</span>
                 {idx < activeLevel ? <CheckCircle size={16} color="#10b981" /> : null}
               </div>
               <h4>{mod.title}</h4>
               <p>{mod.items.join(', ')}</p>
             </div>
           ))}
        </div>

        <div className="control-group mt-auto" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <button 
            className={`toggle-btn ${ttsEnabled ? 'active' : ''}`}
            onClick={() => setTtsEnabled(!ttsEnabled)}
          >
            {ttsEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
            Voice Feedback
          </button>
          <button 
            className="toggle-btn"
            style={{ color: '#ef4444', borderColor: 'rgba(239, 68, 68, 0.2)' }}
            onClick={() => supabase.auth.signOut()}
          >
            <LogOut size={20} />
            Sign Out
          </button>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-stage">
        
        <header className="quiz-header">
          <h2>ISL Interactive Quiz Mode</h2>
          <p>Two-handed spatial recognition active.</p>
        </header>

        {/* HERO AREA */}
        <div className={`quiz-container ${isSuccess ? 'success-pulse' : ''}`}>
           <div className="target-display">
             <span className="target-label">SIGN THIS:</span>
             <h1 className="target-word">{targetSign}</h1>
           </div>

           <div className="video-wrapper">
             <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                className={`webcam ${isSuccess ? 'success-glow' : ''}`}
                videoConstraints={{ facingMode: "user" }}
             />
             
             {!quizActive && targetSign !== "Course Complete!" && (
               <div className="overlay-start">
                 <button className="btn-start" onClick={() => setQuizActive(true)}>
                   <Play size={24} /> Start Challenge
                 </button>
               </div>
             )}

             {/* LIVE CONFIDENCE RING UI */}
             {quizActive && (
               <div className="confidence-hud">
                 <div className="hud-label">AI Confidence</div>
                 <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${confidence}%`, background: confidence > 80 ? '#10b981' : confidence > 40 ? '#f59e0b' : '#ef4444' }}></div>
                 </div>
                 <div className="hud-value">{confidence}%</div>
               </div>
             )}
             
             {isSuccess && (
               <div className="success-overlay">
                 <CheckCircle size={64} color="#10b981" />
                 <h2>Great Job! +{10 + streak*2}</h2>
               </div>
             )}
           </div>
        </div>

      </main>
    </div>
  );
}

export default App;

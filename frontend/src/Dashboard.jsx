import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import { Volume2, VolumeX, Sparkles, Trophy, CheckCircle, Target, ArrowRight, Play, LogOut, LayoutDashboard, Database, Bell, MessageSquare, Settings, HelpCircle, Activity, Book, Sun } from 'lucide-react';
import { supabase } from './supabaseClient';
import './App.css';

const CURRICULUM = [
  { level: 1, title: 'The Basics', items: ['A', 'B', 'C', '1', '2'] },
  { level: 2, title: 'Greetings', items: ['Hello', 'Thank You', 'Please'] },
  { level: 3, title: 'Everyday', items: ['Water', 'Food', 'Stop'] }
];

function App() {
  const webcamRef = useRef(null);
  const [activeView, setActiveView] = useState('quiz');
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
      <aside className="minimal-sidebar">
        <div className="minimal-brand">
          <Sparkles size={24} color="#3b82f6" />
          <h1>ISL Learn</h1>
        </div>

        <div style={{ padding: '0 1.5rem', marginBottom: '1rem', display: 'flex', gap: '1rem', color: '#94a3b8', fontSize: '0.9rem' }}>
           <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}><Trophy size={16} color="#facc15"/> {score}</div>
           <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}><Target size={16} color="#ef4444"/> {streak}x</div>
        </div>

        <nav className="minimal-nav">
          <div className="nav-title" style={{ padding: '0 1.5rem', fontSize: '0.75rem', textTransform: 'uppercase', color: '#64748b', marginBottom: '0.5rem', fontWeight: 600 }}>Curriculum</div>
          <div className={`nav-item ${activeLevel === 0 && activeView === 'quiz' ? 'active' : ''}`} onClick={() => { setActiveView('quiz'); setActiveLevel(0); setTargetIdx(0); setTargetSign(CURRICULUM[0].items[0]); setQuizActive(false); }}>
            <Book size={20} />
            <span>Level 1: The Basics</span>
          </div>
          <div className={`nav-item ${activeLevel === 1 && activeView === 'quiz' ? 'active' : ''}`} onClick={() => { setActiveView('quiz'); setActiveLevel(1); setTargetIdx(0); setTargetSign(CURRICULUM[1].items[0]); setQuizActive(false); }}>
            <MessageSquare size={20} />
            <span>Level 2: Greetings</span>
          </div>
          <div className={`nav-item ${activeLevel === 2 && activeView === 'quiz' ? 'active' : ''}`} onClick={() => { setActiveView('quiz'); setActiveLevel(2); setTargetIdx(0); setTargetSign(CURRICULUM[2].items[0]); setQuizActive(false); }}>
            <Sun size={20} />
            <span>Level 3: Everyday</span>
          </div>

          <div className="nav-title" style={{ padding: '0 1.5rem', fontSize: '0.75rem', textTransform: 'uppercase', color: '#64748b', marginBottom: '0.5rem', marginTop: '1.5rem', fontWeight: 600 }}>Preferences</div>
          <div className={`nav-item ${activeView === 'settings' ? 'active' : ''}`} onClick={() => { setActiveView('settings'); setQuizActive(false); }}>
            <Settings size={20} />
            <span>Account Settings</span>
          </div>
        </nav>

        <div className="minimal-nav-bottom">
          <div className="nav-item" onClick={() => setTtsEnabled(!ttsEnabled)}>
            {ttsEnabled ? <Volume2 size={20} color="#3b82f6" /> : <VolumeX size={20} />}
            <span style={{ color: ttsEnabled ? '#3b82f6' : 'inherit' }}>Voice Feedback {ttsEnabled ? '(On)' : ''}</span>
          </div>
          <div className="nav-item" onClick={() => supabase.auth.signOut()}>
            <LogOut size={20} />
            <span>Sign Out</span>
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="main-stage">
        {activeView === 'settings' ? (
          <div className="settings-view" style={{ animation: 'slideUp 0.5s ease', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            <header className="quiz-header">
              <h2>Account Settings</h2>
              <p>Manage your preferences and profile details securely.</p>
            </header>

            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '2rem', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
               <h3 style={{ marginTop: 0, color: '#fff', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem', fontWeight: 600 }}>Profile Information</h3>
               <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '1.5rem' }}>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: '0.5rem' }}>Email Address</label>
                    <input type="text" value="vvishwakumar19@gmail.com" disabled style={{ width: '100%', padding: '0.8rem 1rem', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#94a3b8', cursor: 'not-allowed', boxSizing: 'border-box' }} />
                  </div>
                  <div>
                    <label style={{ display: 'block', fontSize: '0.85rem', color: '#94a3b8', marginBottom: '0.5rem' }}>Display Name</label>
                    <input type="text" defaultValue="Vishwa Kumar" style={{ width: '100%', padding: '0.8rem 1rem', background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px', color: '#fff', outline: 'none', boxSizing: 'border-box' }} />
                  </div>
               </div>
            </div>

            <div style={{ background: 'rgba(255,255,255,0.02)', padding: '2rem', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
               <h3 style={{ marginTop: 0, color: '#fff', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '1rem', fontWeight: 600 }}>Application Preferences</h3>
               <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginTop: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                     <div>
                       <div style={{ color: '#fff', fontWeight: 500 }}>High Contrast UI</div>
                       <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Improve visibility of UI elements for enhanced accessibility.</div>
                     </div>
                     <button style={{ padding: '0.5rem 1.5rem', borderRadius: '50px', border: '1px solid rgba(255,255,255,0.2)', background: 'transparent', color: '#fff', cursor: 'pointer', fontFamily: 'Inter' }}>Toggle</button>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                     <div>
                       <div style={{ color: '#fff', fontWeight: 500 }}>Experimental AI Model</div>
                       <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Enable the NeuralACT Beta spatial gesture recognition processor.</div>
                     </div>
                     <button style={{ padding: '0.5rem 1.5rem', borderRadius: '50px', border: '1px solid #3b82f6', background: 'rgba(59, 130, 246, 0.1)', color: '#fff', cursor: 'pointer', fontFamily: 'Inter' }}>Enable Mode</button>
                  </div>
               </div>
            </div>

            <div style={{ marginTop: '1rem' }}>
              <button style={{ padding: '0.8rem 2.5rem', background: '#3b82f6', color: '#fff', border: 'none', borderRadius: '8px', fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter', fontSize: '1rem', boxShadow: '0 4px 15px rgba(59, 130, 246, 0.3)' }}>Save Preferences</button>
            </div>
          </div>
        ) : (
          <>
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
                 {quizActive && (
                   <Webcam
                      ref={webcamRef}
                      audio={false}
                      screenshotFormat="image/jpeg"
                      className={`webcam ${isSuccess ? 'success-glow' : ''}`}
                      videoConstraints={{ facingMode: "user" }}
                   />
                 )}
                 
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
          </>
        )}
      </main>
    </div>
  );
}

export default App;

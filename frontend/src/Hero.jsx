import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity } from 'lucide-react';
import './Hero.css';

export default function Hero() {
  const navigate = useNavigate();

  return (
    <div className="hero-container">
      <nav className="hero-nav">
        <div className="brand">
          <Activity size={28} color="#a78bfa" />
          <h1>ISL Learn</h1>
        </div>
        <div className="nav-links">
          <a href="#">Features</a>
          <a href="#">Contacts</a>
          <a href="#">About Us</a>
        </div>
        <div className="nav-actions">
          <button className="btn-nav-login" onClick={() => navigate('/auth')}>Log In</button>
          <button className="btn-nav-signup" onClick={() => navigate('/auth#signup')}>Sign Up</button>
        </div>
      </nav>

      <main className="hero-main">
        <h1 className="hero-title">
          Master your signs.<br />
          <span className="text-glow">One AI platform.</span>
        </h1>
        <p className="hero-subtitle">
          ISL Learn is an all-in-one AI learning platform that<br />
          tracks, analyzes, and scores your Indian Sign Language in real-time.
        </p>
        <button className="btn-hero-start" onClick={() => navigate('/auth')}>
          Get Started
        </button>
      </main>
    </div>
  );
}

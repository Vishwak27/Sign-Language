import React, { useState, useEffect } from 'react';
import { supabase } from './supabaseClient';
import { useNavigate, useLocation } from 'react-router-dom';
import { Activity } from 'lucide-react';
import './Auth.css';

export default function Auth() {
  const location = useLocation();
  const [isLogin, setIsLogin] = useState(!location.hash.includes('signup'));
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [terms, setTerms] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    setIsLogin(!location.hash.includes('signup'));
  }, [location.hash]);

  const handleAuth = async (e) => {
    e.preventDefault();
    if (!isLogin && !terms) {
      setError("Please accept the terms of service.");
      return;
    }

    setLoading(true);
    setError('');

    let result;
    if (isLogin) {
      result = await supabase.auth.signInWithPassword({ email, password });
    } else {
      result = await supabase.auth.signUp({ 
        email, 
        password,
        options: { data: { full_name: name } }
      });
    }

    if (result.error) {
       setError(result.error.message);
    } else {
       navigate('/dashboard');
    }
    setLoading(false);
  };

  const toggleMode = () => {
    navigate(isLogin ? '#signup' : '#login');
  };

  return (
    <div className="auth-container">
      <div className="auth-brand" onClick={() => navigate('/')} style={{ cursor: 'pointer' }}>
        <Activity size={32} color="#a78bfa" />
        <h1>ISL Learn</h1>
      </div>

      <div className="auth-card">
        {isLogin ? (
          <>
            <div className="social-login">
              <button className="btn-social">Log in with GitHub</button>
              <button className="btn-social">Log in with Google</button>
              <button className="btn-social">Log in with SSO</button>
            </div>
            
            <div className="divider"><span>or</span></div>
          </>
        ) : (
          <h2 className="signup-title">Sign up</h2>
        )}

        <form onSubmit={handleAuth}>
          {error && <div className="auth-error">{error}</div>}

          {!isLogin && (
            <div className="input-group">
              <label>Name</label>
              <input type="text" value={name} onChange={e => setName(e.target.value)} required />
            </div>
          )}

          <div className="input-group">
            <div className="label-row">
              <label>Email</label>
              {isLogin && <span className="input-badge">Last used</span>}
            </div>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} required />
          </div>

          <div className="input-group">
            <div className="label-row">
              <label>Password</label>
              {isLogin && <span className="forgot-pass">Forgot password?</span>}
            </div>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required minLength={6} />
          </div>

          {!isLogin && (
            <div className="terms-checkbox">
              <input type="checkbox" id="terms" checked={terms} onChange={e => setTerms(e.target.checked)} />
              <label htmlFor="terms">I accept the <span className="text-link">terms of service</span> and <span className="text-link">privacy policy</span></label>
            </div>
          )}

          <button type="submit" className="btn-auth-submit" disabled={loading}>
            {loading ? 'Processing...' : (isLogin ? 'Log in' : 'Sign up')}
          </button>
        </form>

        <div className="auth-footer">
          {isLogin ? (
            <p>Don't have an account? <span className="text-link" onClick={toggleMode}>Sign up for free!</span></p>
          ) : (
            <p>Already have an account? <span className="text-link" onClick={toggleMode}>Log in!</span></p>
          )}
        </div>
      </div>
    </div>
  );
}

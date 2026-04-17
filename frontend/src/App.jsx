import React, { useEffect, useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { supabase } from './supabaseClient';
import Hero from './Hero';
import Auth from './Auth';
import Dashboard from './Dashboard';
import './App.css';

export default function App() {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setLoading(false);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });

    return () => subscription.unsubscribe();
  }, []);

  if (loading) return null; // Or a spinner

  return (
    <Router>
       <Routes>
          <Route path="/" element={!session ? <Hero /> : <Navigate to="/dashboard" />} />
          <Route path="/auth" element={!session ? <Auth /> : <Navigate to="/dashboard" />} />
          <Route path="/dashboard" element={session ? <Dashboard /> : <Navigate to="/auth" />} />
       </Routes>
    </Router>
  );
}

import React, { useEffect, useState } from 'react'
import { Link, NavLink, Navigate, Route, Routes, useLocation, useNavigate, useParams } from 'react-router-dom'
import { RunListView } from './views/RunListView'
import { RunDetailView } from './views/RunDetailView'
import { CompareView } from './views/CompareView'
import { ArenaView } from './views/ArenaView'
import { ModelsView } from './views/ModelsView'
import { ReplaysView } from './views/ReplaysView'
import { LabView, PlayView } from './views/Transitions'
import { runRoute } from './platform'
import { ResearchContext } from './components/ResearchContext'
import './App.css'

function RunPage() {
  const { runId = '' } = useParams()
  const navigate = useNavigate()
  return <RunDetailView key={runId} runId={runId} onBack={() => navigate('/training')} />
}
function LegacyRun() {
  const { runId = '' } = useParams()
  const { search } = useLocation()
  return <Navigate replace to={runRoute(runId) + search} />
}
function TrainingPage() {
  const navigate = useNavigate()
  return <><div className="page-heading"><div><p className="eyebrow">Research / Training</p><h1>Training runs</h1><p className="muted">Legacy and MLflow evidence in one workspace.</p></div><Link className="button-link" to="/training/compare">Compare runs</Link></div><RunListView onSelectRun={id => navigate(runRoute(id))} /></>
}
export default function App() {
  const navigate = useNavigate()
  const location = useLocation()
  const [theme, setTheme] = useState(() => {
    try { const saved = localStorage.getItem('quoridor-theme'); if (saved === 'light' || saved === 'dark') return saved } catch { /* Storage may be disabled. */ }
    return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })
  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem('quoridor-theme', theme) } catch { /* Nonessential preference. */ }
  }, [theme])
  useEffect(() => { document.title = `Quoridor · ${location.pathname.split('/')[1] || 'Research'}` }, [location.pathname])
  return <div className="app">
    <a className="skip-link" href="#main-content">Skip to content</a>
    <header className="app-header">
      <Link className="header-title" to="/training"><span className="brand-mark">Q</span> Quoridor <small>Research</small></Link>
      <nav className="header-nav" aria-label="Main navigation">
        {['Play', 'Training', 'Arena', 'Replays', 'Lab', 'Models'].map(label => <NavLink key={label} to={`/${label.toLowerCase()}`}>{label === 'Replays' ? 'Replay' : label}{label === 'Play' && <span aria-label="external transition"> ↗</span>}</NavLink>)}
      </nav>
      <span className="header-badge">Read-only research</span>
      <button aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} theme`} onClick={() => setTheme(t => t === 'light' ? 'dark' : 'light')}>{theme === 'light' ? '☾ Dark' : '☀ Light'}</button>
    </header>
    <main id="main-content" className="app-main" tabIndex={-1}>
      {location.pathname !== '/lab' && !location.pathname.startsWith('/replays/') && <ResearchContext />}
      <Routes>
        <Route path="/" element={<Navigate replace to={location.hash === '#arena' ? '/arena' : '/training'} />} />
        <Route path="/training" element={<TrainingPage />} />
        <Route path="/training/runs/:runId" element={<RunPage />} />
        <Route path="/training/compare" element={<CompareView onBack={() => navigate('/training')} />} />
        <Route path="/arena" element={<ArenaView />} />
        <Route path="/replays" element={<ReplaysView />} />
        <Route path="/replays/:runId" element={<ReplaysView />} />
        <Route path="/models" element={<ModelsView />} />
        <Route path="/models/:sha" element={<ModelsView />} />
        <Route path="/lab" element={<LabView />} />
        <Route path="/play" element={<PlayView />} />
        <Route path="/run/:runId" element={<LegacyRun />} />
        <Route path="/compare" element={<Navigate replace to={'/training/compare' + location.search} />} />
        <Route path="*" element={<div className="empty"><h1>Page not found</h1><Link to="/training">Return to Training</Link></div>} />
      </Routes>
    </main>
    <footer className="app-footer">Immutable identities · Recorded trajectories · Engine-authoritative positions</footer>
  </div>
}

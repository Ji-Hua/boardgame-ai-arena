import React, { useState } from 'react'
import { RunListView } from './views/RunListView'
import { RunDetailView } from './views/RunDetailView'
import { CompareView } from './views/CompareView'
import { ArenaView } from './views/ArenaView'
import './App.css'

type View =
  | { kind: 'list' }
  | { kind: 'detail'; runId: string }
  | { kind: 'compare' }
  | { kind: 'arena' }

export default function App() {
  const [view, setView] = useState<View>({ kind: window.location.hash === '#arena' ? 'arena' : 'list' })

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-title" onClick={() => setView({ kind: 'list' })} style={{ cursor: 'pointer' }}>
          Quoridor Training Dashboard
        </div>
        <nav className="header-nav">
          <button
            className={view.kind === 'arena' ? 'active' : ''}
            onClick={() => { window.location.hash = 'arena'; setView({ kind: 'arena' }) }}
          >
            Arena
          </button>
          <button
            className={view.kind === 'list' ? 'active' : ''}
            onClick={() => setView({ kind: 'list' })}
          >
            Runs
          </button>
          <button
            className={view.kind === 'compare' ? 'active' : ''}
            onClick={() => setView({ kind: 'compare' })}
          >
            Compare
          </button>
        </nav>
        <div className="header-badge">Read-Only</div>
      </header>

      <main className="app-main">
        {view.kind === 'arena' && <ArenaView />}
        {view.kind === 'list' && (
          <RunListView onSelectRun={runId => setView({ kind: 'detail', runId })} />
        )}
        {view.kind === 'detail' && (
          <RunDetailView
            runId={view.runId}
            onBack={() => setView({ kind: 'list' })}
          />
        )}
        {view.kind === 'compare' && (
          <CompareView onBack={() => setView({ kind: 'list' })} />
        )}
      </main>
    </div>
  )
}

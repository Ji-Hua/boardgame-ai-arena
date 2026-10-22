import React, { useEffect, useState } from 'react'
import { LineChart } from '../components/LineChart'

type Experiment = { id: string; name: string }
type Run = { run_id: string; run_name: string | null; status: string; params: Record<string, string>; tags: Record<string, string>; metrics: Record<string, number> }
type HistoryPoint = { step: number; value: number; timestamp: number }
type Evaluation = { available: boolean; checkpoint_sha256?: string; agent_id?: string; summary?: { games: number; wins: number; draws: number; losses: number; score: number }; reason?: string; game_records_reason?: string }

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path)
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<T>
}

export function MLflowView() {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [selected, setSelected] = useState<string>('')
  const [runs, setRuns] = useState<Run[]>([])
  const [error, setError] = useState<string | null>(null)
  const [history, setHistory] = useState<{ run: Run; key: string; points: HistoryPoint[] } | null>(null)
  const [evaluations, setEvaluations] = useState<Record<string, Evaluation>>({})

  useEffect(() => {
    get<{ experiments: Experiment[] }>('/api/mlflow/experiments')
      .then(data => { setExperiments(data.experiments); setSelected(data.experiments[0]?.id ?? '') })
      .catch(error => setError(String(error)))
  }, [])
  useEffect(() => {
    if (!selected) return
    get<{ runs: Run[] }>(`/api/mlflow/experiments/${encodeURIComponent(selected)}/runs`)
      .then(data => setRuns(data.runs)).catch(error => setError(String(error)))
  }, [selected])
  if (error) return <div className="error">MLflow unavailable: {error}</div>
  const showHistory = (run: Run, key: string) => get<{ metrics: HistoryPoint[] }>(
    `/api/mlflow/runs/${encodeURIComponent(run.run_id)}/metrics/${encodeURIComponent(key)}`,
  ).then(data => setHistory({ run, key, points: data.metrics })).catch(error => setError(String(error)))
  const showEvaluation = (run: Run) => get<Evaluation>(
    `/api/mlflow/runs/${encodeURIComponent(run.run_id)}/arena-evaluation`,
  ).then(data => setEvaluations(current => ({ ...current, [run.run_id]: data }))).catch(error => setError(String(error)))
  return <div>
    <h2>MLflow Experiments</h2>
    <p className="muted">Read-only Tracking metadata. Metrics appear only when MLflow recorded them; policy logits, Q-values, rewards and values remain distinct.</p>
    <select value={selected} onChange={event => setSelected(event.target.value)}>
      {experiments.map(experiment => <option key={experiment.id} value={experiment.id}>{experiment.name}</option>)}
    </select>
    <div className="table-scroll">
      <table><thead><tr><th>Run</th><th>Status</th><th>Algorithm</th><th>Checkpoint SHA</th><th>Available metrics</th><th>Source</th></tr></thead>
        <tbody>{runs.map(run => <React.Fragment key={run.run_id}><tr>
          <td className="mono">{run.run_name ?? run.run_id}</td><td>{run.status}</td>
          <td>{run.tags.algorithm ?? run.params.algorithm ?? '—'}</td>
          <td className="mono">{run.tags.checkpoint_sha256?.slice(0, 12) ?? '—'}</td>
          <td>{Object.keys(run.metrics).length
            ? Object.keys(run.metrics).map(key => <button key={key} onClick={() => showHistory(run, key)}>{key}</button>)
            : 'No MLflow metrics recorded'}</td>
          <td className="mono">{run.params.source_path ?? 'MLflow artifact only'}</td>
        </tr><tr><td colSpan={6}><button onClick={() => showEvaluation(run)}>Arena evaluation</button>
          {evaluations[run.run_id] && <span className="muted"> {evaluations[run.run_id].available
            ? `${evaluations[run.run_id].agent_id}: ${evaluations[run.run_id].summary?.games} games — ${evaluations[run.run_id].summary?.wins}W / ${evaluations[run.run_id].summary?.draws}D / ${evaluations[run.run_id].summary?.losses}L; checkpoint ${evaluations[run.run_id].checkpoint_sha256?.slice(0, 12)}. ${evaluations[run.run_id].game_records_reason}`
            : evaluations[run.run_id].reason}</span>}</td></tr></React.Fragment>)}</tbody>
      </table>
    </div>
    {history && <div className="chart-card"><h3>{history.run.run_name ?? history.run.run_id} — {history.key}</h3>
      {history.points.length ? <LineChart points={history.points} xKey="step" yKey="value" color="#4f9eff" width={700} height={240} /> : <p className="muted">MLflow has no history for this metric.</p>}
    </div>}
  </div>
}

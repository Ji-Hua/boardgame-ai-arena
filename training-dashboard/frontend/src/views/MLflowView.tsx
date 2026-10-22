import React, { useEffect, useState } from 'react'

type Experiment = { id: string; name: string }
type Run = { run_id: string; run_name: string | null; status: string; params: Record<string, string>; tags: Record<string, string>; metrics: Record<string, number> }

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
  return <div>
    <h2>MLflow Experiments</h2>
    <p className="muted">Read-only Tracking metadata. Metrics appear only when MLflow recorded them; policy logits, Q-values, rewards and values remain distinct.</p>
    <select value={selected} onChange={event => setSelected(event.target.value)}>
      {experiments.map(experiment => <option key={experiment.id} value={experiment.id}>{experiment.name}</option>)}
    </select>
    <div className="table-scroll">
      <table><thead><tr><th>Run</th><th>Status</th><th>Algorithm</th><th>Checkpoint SHA</th><th>Available metrics</th><th>Source</th></tr></thead>
        <tbody>{runs.map(run => <tr key={run.run_id}>
          <td className="mono">{run.run_name ?? run.run_id}</td><td>{run.status}</td>
          <td>{run.tags.algorithm ?? run.params.algorithm ?? '—'}</td>
          <td className="mono">{run.tags.checkpoint_sha256?.slice(0, 12) ?? '—'}</td>
          <td>{Object.keys(run.metrics).join(', ') || 'No MLflow metrics recorded'}</td>
          <td className="mono">{run.params.source_path ?? 'MLflow artifact only'}</td>
        </tr>)}</tbody>
      </table>
    </div>
  </div>
}

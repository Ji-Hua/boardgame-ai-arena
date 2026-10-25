import React, { useEffect, useState } from 'react'
import { api } from '../api'
import type { PlatformModel } from '../platform'

export function ModelPicker({ value, onChange, capability = 'analyzable', recordedSha }: {
  value: string; onChange: (sha: string) => void; capability?: 'analyzable' | 'playable'; recordedSha?: string | null
}) {
  const [models, setModels] = useState<PlatformModel[]>([])
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    api.models().then(data => { if (active) setModels(data.models) }).catch(e => { if (active) setError(String(e)) })
    return () => { active = false }
  }, [])
  const selected = models.find(model => model.sha256 === value)
  return <div className="model-picker">
    <label>Analysis model (recorded checkpoint or hypothetical override)
      <select aria-label="Analysis model" value={value} onChange={event => onChange(event.target.value)}>
        <option value="">{recordedSha ? `Use recorded model · ${recordedSha}` : 'Resolve recorded model from source (if available); otherwise no model'}</option>
        {value && !selected && <option value={value} disabled>{value} · capability not yet verified</option>}
        {models.map(model => <option key={model.sha256} value={model.sha256} disabled={!model[capability]}>
          {model.aliases.join(' / ') || model.architecture || 'Model'} · {model.sha256}{!model[capability] ? ` · Disabled: ${(model.reasons[capability] ?? []).join(', ')}` : ''}
        </option>)}
      </select>
    </label>
    {recordedSha && (!value || value === recordedSha) && <p className="muted sha">Analysis uses the recorded model · {recordedSha}. Server validation is still required.</p>}
    {error && <p className="warning">Model capabilities unavailable: {error}. No eligibility is assumed.</p>}
    {selected && <p className={selected[capability] ? 'muted' : 'warning'}>{selected[capability] ? 'Static eligibility only; successful loading is not verified.' : 'Disabled.'} {(selected.reasons[capability] ?? []).join(' · ')}</p>}
    {value && !selected && <p className="warning">Requested SHA has no loaded capability evidence. Resolve it in Models before choosing it for analysis.</p>}
    <details><summary>Capabilities and disabled reasons</summary>{models.map(model => <div className="capability-row" key={model.sha256}><span className="mono">{model.sha256}</span><span>{model[capability] ? 'Eligible (static)' : 'Disabled'} · {(model.reasons[capability] ?? []).join(' · ')}</span></div>)}</details>
  </div>
}
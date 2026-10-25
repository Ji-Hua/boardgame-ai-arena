import React, { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { runRoute, replayRoute } from '../platform'
import EvalReplaysView from './EvalReplaysView'

export function ArenaView() {
  const [params] = useSearchParams()
  const runId = params.get('run_id')
  return <div>
    <div className="page-heading"><div><p className="eyebrow">Research / Arena</p><h1>{runId ? 'Run Arena evaluations' : 'Arena evaluation evidence'}</h1><p className="muted">Exact checkpoint identities, recorded protocols and aggregate results. Replay samples are not aggregate scores.</p></div><div className="action-bar"><Link to="/arena">Latest Arena matrix</Link><Link to="/training">Find a training run</Link></div></div>
    {runId ? <><div className="action-bar"><Link to={runRoute(runId, 'checkpoints')}>Run checkpoints</Link><Link to={replayRoute(runId, { checkpoint_sha: params.get('checkpoint_sha'), checkpoint: params.get('checkpoint'), evaluation: params.get('evaluation') })}>Replay archive</Link></div><h2>Recorded evaluations &amp; replay evidence</h2><EvalReplaysView key={runId} runId={runId} arena /></> : <LatestArenaView />}
  </div>
}

type Cell = {
  games: number; wins: number; losses: number; draws: number; score: number | null
  ci95: number[]; p1_score: number | null; p2_score: number | null
  p1_games: number; p2_games: number; terminations: Record<string, number>
}
type Rank = Cell & { id: string; rank: number; elo: number; mean_decision_ms: number; forfeits: number }
type Report = {
  run_id: string; status: string; completed_blocks: number; expected_blocks: number
  ranked_games: number; control_games: number; pairs_per_match: number
  unique_openings: number; max_plies: number; updated_at: string
  source_sha256: string; manifest_sha256: string
  agents: { id: string }[]; ranking: Rank[]; matrix: Record<string, Record<string, Cell | null>>
  terminations: Record<string, number>; forfeits: unknown[]
  standard_controls: { a: string; b: string; a_score: number; terminations: string[] }[]
}
type Catalog = {
  models: { id: string; sha256: string; selection_reason: string; historical_training_commit: string
    metadata: { episode_count: number; observation_version: string; model_config: { model_arch?: string } } }[]
  registry: { id: string; name: string; version: string; run_id: string }[]
}
const labels: Record<string, string> = {
  random_legal: '随机', greedy: 'Greedy', minimax_d1: 'Minimax d1', minimax_d2: 'Minimax d2',
  mlp_stable: 'MLP stable', cnn_random_2k: 'CNN random 2k', cnn_observation_2k: 'CNN obs 2k',
  cnn_curriculum_peak: 'CNN curriculum', cnn_mixed_peak: 'CNN mixed', cnn_5k_final: 'CNN 5k final',
}
const name = (id: string) => labels[id] || id
const pct = (v: number | null | undefined) => v == null ? '—' : `${(100 * v).toFixed(1)}%`
const interval = (ci: number[]) => `${pct(ci[0])} – ${pct(ci[1])}`
const reasons: Record<string, string> = {
  goal: '到达终点', deterministic_cycle: '确定性循环', ply_limit: '达到步数上限',
  illegal_action: '非法出招', agent_error: '推理异常',
}

function LatestArenaView() {
  const [report, setReport] = useState<Report>()
  const [catalog, setCatalog] = useState<Catalog>()
  const [error, setError] = useState('')
  const [modelError, setModelError] = useState('')
  const [selected, setSelected] = useState<[string, string]>()
  const mlflow = `${window.location.protocol}//${window.location.hostname}:8742`
  useEffect(() => {
    let active = true
    const refresh = async () => {
      try {
        const response = await fetch('/api/arena/latest')
        if (!response.ok) throw new Error(`对战数据暂不可用 (${response.status})`)
        const data = await response.json()
        if (active) { setReport(data); setError('') }
      } catch (err) { if (active) setError(String(err)) }
    }
    void refresh()
    const timer = window.setInterval(refresh, 10000)
    fetch('/api/models').then(async response => {
      if (!response.ok) throw new Error(`模型库暂不可用 (${response.status})`)
      const data = await response.json()
      if (active) setCatalog(data)
    }).catch(err => { if (active) setModelError(String(err)) })
    return () => { active = false; window.clearInterval(timer) }
  }, [])
  if (!report) return <div className={error ? 'error' : 'loading'}>{error || '正在读取对战矩阵…'}</div>
  const chosen = selected && report.matrix[selected[0]]?.[selected[1]]
  const control = selected && report.standard_controls.find(c =>
    [c.a, c.b].includes(selected[0]) && [c.a, c.b].includes(selected[1]))
  return <div className="arena-view">
    <div className="arena-heading">
      <div><h2>对战矩阵与模型库</h2><p>{report.run_id} · {report.status === 'complete' ? '已完成' : '评测中，每 10 秒更新'}</p></div>
      <div className="arena-links"><Link to="/play">External Play ↗</Link><Link to="/replays">Recorded replays</Link><Link to="/models">Model identities</Link><a href={mlflow + '/#/models'} target="_blank" rel="noreferrer">MLflow ↗</a></div>
    </div>
    {error && <div className="warning">{error}；以下为上次成功读取的数据。</div>}
    {report.forfeits.length > 0 && <div className="error">{report.forfeits.length} 局发生弃权，需检查出招或推理错误后再解释排名。</div>}
    <div className="cards-row">
      <div className="card"><div className="card-label">参赛 agent</div><div className="card-value">{report.agents.length}</div></div>
      <div className="card"><div className="card-label">排名对局 / 标准开局对照</div><div className="card-value">{report.ranked_games.toLocaleString()} / {report.control_games}</div></div>
      <div className="card"><div className="card-label">每组计划局数 · 先后手对半</div><div className="card-value">{report.pairs_per_match * 2}</div></div>
      <div className="card"><div className="card-label">评测进度 · 两局为一组</div><div className="card-value">{report.completed_blocks} / {report.expected_blocks}</div></div>
    </div>
    <div className="arena-note">
      得分率 =（胜 + ½ 和）/ 局数。排名来自 {report.pairs_per_match} 个随机合法开局，每个开局交换先后手各打一局；
      标准初始局面的对照另列。95% 区间按配对开局计算，50 局仅作初筛，接近的名次不能据此认定强弱。
      Elo 以随机 agent = 1500 为基准，仅适用于本轮参赛池。
    </div>
    <section>
      <h3>排名 · 按总得分率排序</h3>
      <div className="table-scroll"><table>
        <thead><tr><th>#</th><th>Agent</th><th>胜 / 和 / 负</th><th>得分率</th><th>95% 保守区间</th><th>Elo · 暂定</th><th>执先 / 执后</th><th>平均决策</th></tr></thead>
        <tbody>{report.ranking.map(row => <tr key={row.id}>
          <td>{row.rank}</td><td title={row.id}>{name(row.id)}</td><td>{row.wins} / {row.draws} / {row.losses}</td>
          <td className="good">{pct(row.score)}</td><td>{interval(row.ci95)}</td><td>{Math.round(row.elo)}</td>
          <td>{pct(row.p1_score)} / {pct(row.p2_score)}</td><td>{row.mean_decision_ms.toFixed(2)} ms</td>
        </tr>)}</tbody>
      </table></div>
    </section>
    <section>
      <h3>交叉矩阵 · 行 agent 对列 agent</h3>
      <p className="arena-hint">点击格子查看先后手、胜负和及标准开局对照。</p>
      <div className="table-scroll"><table className="arena-matrix">
        <thead><tr><th>得分率</th>{report.agents.map((a, i) => <th key={a.id} title={name(a.id)}>{i + 1}</th>)}</tr></thead>
        <tbody>{report.agents.map((a, i) => <tr key={a.id}>
          <th>{i + 1}. {name(a.id)}</th>{report.agents.map(b => {
            const cell = report.matrix[a.id][b.id]
            return <td key={b.id}>{cell ? <button
              aria-label={`${name(a.id)} 对 ${name(b.id)}`}
              className={selected?.[0] === a.id && selected?.[1] === b.id ? 'selected' : ''}
              style={{ background: cell.score == null ? undefined : `hsl(${cell.score * 125} 38% 19%)` }}
              onClick={() => setSelected([a.id, b.id])}
            >{pct(cell.score)}<small>{cell.games} 局</small></button> : '—'}</td>
          })}
        </tr>)}</tbody>
      </table></div>
      {selected && chosen && <div className="arena-detail" aria-live="polite">
        <strong>{name(selected[0])} → {name(selected[1])}</strong>
        <p>{chosen.wins} 胜 / {chosen.draws} 和 / {chosen.losses} 负 · 得分率 {pct(chosen.score)} · 95% 区间 {interval(chosen.ci95)}</p>
        <p>执先 {chosen.p1_games} 局：{pct(chosen.p1_score)}；执后 {chosen.p2_games} 局：{pct(chosen.p2_score)}。</p>
        <p>{Object.entries(chosen.terminations).map(([key, count]) => `${reasons[key] || key} ${count} 局`).join(' · ')}</p>
        {control && <p>标准初始局面两局对照：{name(selected[0])} 得分率 {pct(control.a === selected[0] ? control.a_score : 1 - control.a_score)}（不计入排名）。</p>}
      </div>}
    </section>
    <section>
      <h3>模型版本 · 固定 checkpoint</h3>
      {modelError && <div className="warning">{modelError}</div>}
      <div className="table-scroll"><table>
        <thead><tr><th>模型</th><th>训练局数</th><th>编码器</th><th>SHA256</th><th>Registry</th></tr></thead>
        <tbody>{catalog?.models.map(model => {
          const version = catalog.registry.find(v => v.id === model.id)
          return <tr key={model.id}>
            <td title={model.selection_reason}>{name(model.id)}</td><td>{model.metadata.episode_count}</td>
            <td>{model.metadata.observation_version}</td><td className="mono" title={model.sha256}>{model.sha256.slice(0, 16)}</td>
            <td>{version ? <a href={`${mlflow}/#/models/${encodeURIComponent(version.name)}/versions/${version.version}`} target="_blank" rel="noreferrer">{version.name} · v{version.version} ↗</a> : '登记中'}</td>
          </tr>
        })}</tbody>
      </table></div>
      <p className="arena-hint">历史 checkpoint 的训练代码版本未知；本轮评测源码和文件哈希已固定。登记与测评不自动授予 champion。</p>
    </section>
    <details className="arena-detail">
      <summary>评测协议与可追溯信息</summary>
      <p>唯一开局数：{report.unique_openings}；步数上限：{report.max_plies}；终止原因：{Object.entries(report.terminations).map(([k, n]) => `${reasons[k] || k} ${n}`).join(' · ')}。</p>
      <p>置信区间：固定样本量 Hoeffding 界，采样单元为交换先后手的两局；单项 95%，未做多重比较校正。观察到的名次不等于显著差异。</p>
      <p>确定性、无记忆策略出现完整状态循环时计和；其余对局达到步数上限计和。</p>
      <p className="mono">Manifest: {report.manifest_sha256}<br />Source: {report.source_sha256}</p>
      <p><a href={`/api/arena/runs/${report.run_id}/manifest`} target="_blank" rel="noreferrer">查看模型、源码摘要与开局清单 JSON</a> · <a href="/api/arena/latest" target="_blank" rel="noreferrer">查看完整结果 JSON</a></p>
      <p>最近更新：{new Date(report.updated_at).toLocaleString()}</p>
    </details>
  </div>
}

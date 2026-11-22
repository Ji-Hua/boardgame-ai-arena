"""Training health alerts computed from a Job's recorded metric and evaluation rows.

Rules only read numbers the run already wrote, so they apply to any entrypoint that
records the field; a missing field disables its rule. An alert asks the researcher to
look at the run, it never stops a Job.
"""
from __future__ import annotations

import math
from statistics import median

FINITE_KEYS = ('loss', 'q_mean', 'avg_q_max', 'grad_norm', 'td_error_max')
TRUNCATION_RATE = 0.3     # share of recent episodes that hit the step limit
REPEAT_RATE = 0.25        # share of learner decisions in an already-seen position
Q_MAGNITUDE = 5.0         # terminal rewards are +-1, so |Q| far above 1 means divergence
LOSS_SPIKE = 10.0         # latest loss over the median of earlier rows
EVAL_DROP = 0.2           # win-rate drop from the best earlier evaluation, per opponent
STALLED_SECONDS = 600     # a running Job without new metrics


def _alert(level, code, message, row=None):
    return dict(level=level, code=code, message=message, episode=(row or {}).get('episode'))


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def assess(rows, evaluations=(), *, running=False, last_write=None, now=None):
    rows = [r for r in rows if r.get('type', 'train') == 'train']
    alerts = []
    if running and last_write is not None and now is not None and now - last_write > STALLED_SECONDS:
        alerts.append(_alert('warning', 'stalled', f'{int((now - last_write) // 60)} 分钟没有新的训练指标'))
    if rows:
        last = rows[-1]
        broken = [k for k in FINITE_KEYS if _number(last.get(k)) and not math.isfinite(last[k])]
        if broken:
            alerts.append(_alert('error', 'non_finite', '出现 NaN / Inf：' + '、'.join(broken), last))
        if _number(last.get('truncation_rate')) and last['truncation_rate'] >= TRUNCATION_RATE:
            alerts.append(_alert('warning', 'truncation',
                                 f"最近 {last['truncation_rate']:.0%} 的对局达到步数上限，可能在绕圈", last))
        if _number(last.get('repeat_rate')) and last['repeat_rate'] >= REPEAT_RATE:
            alerts.append(_alert('warning', 'repetition', f"{last['repeat_rate']:.0%} 的决策回到本局走过的局面", last))
        q = last.get('avg_q_max')
        if _number(q) and math.isfinite(q) and abs(q) > Q_MAGNITUDE:
            alerts.append(_alert('warning', 'q_divergence', f'|Q| 均值 {q:.2f}，远超奖励范围 ±1，价值可能发散', last))
        earlier = [r['loss'] for r in rows[:-1] if _number(r.get('loss')) and math.isfinite(r['loss'])]
        loss = last.get('loss')
        if len(earlier) >= 5 and _number(loss) and math.isfinite(loss) and median(earlier) > 0 and loss > LOSS_SPIKE * median(earlier):
            alerts.append(_alert('warning', 'loss_spike', f'loss {loss:.4g} 是此前中位数的 {loss / median(earlier):.0f} 倍', last))
    by_opponent = {}
    for row in evaluations:
        name = row.get('opponent_name') or row.get('opponent') or row.get('opponent_id')
        if name and _number(row.get('win_rate')):
            by_opponent.setdefault(name, []).append(row)
    for name, history in by_opponent.items():
        best = max(history[:-1], key=lambda r: r['win_rate'], default=None)
        if best and history[-1]['win_rate'] <= best['win_rate'] - EVAL_DROP:
            alerts.append(_alert('warning', 'eval_regression',
                                 f"对 {name} 胜率从 {best['win_rate']:.0%}（episode {best.get('episode')}）降到 {history[-1]['win_rate']:.0%}",
                                 history[-1]))
    return alerts

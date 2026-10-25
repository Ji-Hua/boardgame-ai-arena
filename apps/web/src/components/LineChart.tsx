import React, { useState } from 'react'
import type { AxisKey, MetricPoint } from '../types'
import { chartData, chartDomains } from '../runData'

export interface ChartSeries {
  points: MetricPoint[]
  yKey: string
  label?: string
  color?: string
  xKey?: AxisKey
}

interface LineChartProps {
  points?: MetricPoint[]
  series?: ChartSeries[]
  xKey: AxisKey
  xLabel?: string
  yKey?: string
  yLabel?: string
  connect?: boolean
  color?: string
  label?: string
  width?: number
  height?: number
}

interface TooltipState {
  visible: boolean
  x: number
  y: number
  xVal: number
  yVal: number
  label?: string
}

export function LineChart({ points = [], series, xKey, xLabel, yKey = '', yLabel, connect = true, color = '#4f9eff', label, width = 600, height = 200 }: LineChartProps) {
  const [tooltip, setTooltip] = useState<TooltipState>({ visible: false, x: 0, y: 0, xVal: 0, yVal: 0 })
  const pad = { top: 10, right: 10, bottom: 30, left: 55 }

  const curves = (series ?? [{ points, yKey, color, label }]).map(s => ({ ...s, data: chartData(s.points, s.xKey ?? xKey, s.yKey) }))
  const data = curves.flatMap(s => s.data.map(p => ({ ...p, label: s.label })))
  const domains = chartDomains(series ?? [{ points, yKey }], xKey)

  if (!domains || data.length === 0) {
    return (
      <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666', fontSize: 12 }}>
        No data for {yKey}
      </div>
    )
  }

  let { xMin, xMax, yMin, yMax } = domains
  if (xMin === xMax) { xMin -= 0.5; xMax += 0.5 }
  if (yMin === yMax) { const delta = Math.abs(yMin) * 0.05 || 0.5; yMin -= delta; yMax += delta }

  const chartW = width - pad.left - pad.right
  const chartH = height - pad.top - pad.bottom

  const xRange = xMax - xMin || 1
  const yRange = yMax - yMin || 1

  const scaleX = (x: number) => ((x - xMin) / xRange) * chartW
  const scaleY = (y: number) => chartH - ((y - yMin) / yRange) * chartH

  const yTicks = 4
  const xTicks = 5

  function handleMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    const svgRect = e.currentTarget.getBoundingClientRect()
    const mouseX = (e.clientX - svgRect.left) * width / svgRect.width - pad.left
    const mouseY = (e.clientY - svgRect.top) * height / svgRect.height - pad.top

    if (mouseX < 0 || mouseX > chartW || mouseY < 0 || mouseY > chartH) {
      setTooltip(t => ({ ...t, visible: false }))
      return
    }

    // Find nearest data point by x
    let nearest = data[0]
    let nearestDist = Infinity
    for (const d of data) {
      const dist = Math.hypot(scaleX(d.x) - mouseX, scaleY(d.y) - mouseY)
      if (dist < nearestDist) {
        nearestDist = dist
        nearest = d
      }
    }

    setTooltip({
      visible: true,
      x: e.clientX + 12,
      y: e.clientY - 10,
      xVal: nearest.x,
      yVal: nearest.y,
      label: nearest.label,
    })
  }

  function handleMouseLeave() {
    setTooltip(t => ({ ...t, visible: false }))
  }

  const fmtY = (v: number) => {
    if (Math.abs(v) >= 1000) return v.toFixed(0)
    if (Math.abs(v) < 0.01 && v !== 0) return v.toExponential(2)
    return v.toFixed(3)
  }

  return (
    <>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`${yLabel ?? yKey} by ${xLabel ?? xKey}`}
        style={{ display: 'block', cursor: 'crosshair', width: '100%', maxWidth: width }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <g className="chart-plot" transform={`translate(${pad.left},${pad.top})`}>
          {/* Grid lines */}
          {Array.from({ length: yTicks + 1 }, (_, i) => {
            const v = yMin + (yRange * i) / yTicks
            const y = scaleY(v)
            return (
              <g key={i}>
                <line x1={0} y1={y} x2={chartW} y2={y} stroke="#333" strokeWidth={0.5} />
                <text x={-4} y={y + 4} textAnchor="end" fontSize={10} fill="#aaa">
                  {fmtY(v)}
                </text>
              </g>
            )
          })}
          {Array.from({ length: xTicks + 1 }, (_, i) => {
            const v = xMin + (xRange * i) / xTicks
            const x = scaleX(v)
            return (
              <g key={i}>
                <line x1={x} y1={0} x2={x} y2={chartH} stroke="#333" strokeWidth={0.5} />
                <text x={x} y={chartH + 15} textAnchor="middle" fontSize={10} fill="#aaa">
                  {xKey === 'timestamp' ? new Date(v).toLocaleDateString() : Math.round(v)}
                </text>
              </g>
            )
          })}
          {/* Data line */}
          {curves.map((curve, index) => <g key={index}>
            {connect && <path d={curve.data.map((p, i) => `${i ? 'L' : 'M'} ${scaleX(p.x)} ${scaleY(p.y)}`).join(' ')} fill="none" stroke={curve.color ?? color} strokeWidth={1.5} />}
            {curve.data.map((p, i) => <circle key={i} cx={scaleX(p.x)} cy={scaleY(p.y)} r={2} fill={curve.color ?? color} />)}
          </g>)}
          {/* Hover indicator */}
          {tooltip.visible && (
            <line
              x1={scaleX(tooltip.xVal)}
              y1={0}
              x2={scaleX(tooltip.xVal)}
              y2={chartH}
              stroke={color}
              strokeWidth={1}
              strokeDasharray="3,3"
              opacity={0.7}
            />
          )}
          {/* Axes */}
          <line x1={0} y1={0} x2={0} y2={chartH} stroke="#555" strokeWidth={1} />
          <line x1={0} y1={chartH} x2={chartW} y2={chartH} stroke="#555" strokeWidth={1} />
          {/* X-axis label */}
          <text x={chartW / 2} y={chartH + 28} textAnchor="middle" fontSize={10} fill="#888">
            {xLabel ?? xKey}
          </text>
          {/* Legend */}
          {label && (
            <text x={4} y={14} fontSize={11} fill={color}>{label}</text>
          )}
        </g>
      </svg>
      {tooltip.visible && (
        <div
          className="chart-tooltip"
          style={{ left: tooltip.x, top: tooltip.y }}
        >
          {tooltip.label && <strong>{tooltip.label} · </strong>}
          <span style={{ color: '#888' }}>{xLabel ?? xKey}:</span> {xKey === 'timestamp' ? new Date(tooltip.xVal).toLocaleString() : tooltip.xVal}&nbsp;&nbsp;
          <span style={{ color: '#888' }}>{yLabel ?? yKey}:</span> {fmtY(tooltip.yVal)}
        </div>
      )}
    </>
  )
}

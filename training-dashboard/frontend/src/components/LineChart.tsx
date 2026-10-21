import React, { useState } from 'react'
import type { MetricPoint } from '../types'

interface LineChartProps {
  points: MetricPoint[]
  xKey: 'episode' | 'step'
  yKey: string
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
}

function toNum(v: number | null | undefined): number | null {
  if (v == null || isNaN(v)) return null
  return v
}

export function LineChart({ points, xKey, yKey, color = '#4f9eff', label, width = 600, height = 200 }: LineChartProps) {
  const [tooltip, setTooltip] = useState<TooltipState>({ visible: false, x: 0, y: 0, xVal: 0, yVal: 0 })
  const pad = { top: 10, right: 10, bottom: 30, left: 55 }

  const data = points
    .map(p => ({ x: toNum(p[xKey]), y: toNum(p.values[yKey]) }))
    .filter((d): d is { x: number; y: number } => d.x != null && d.y != null)

  if (data.length === 0) {
    return (
      <div style={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666', fontSize: 12 }}>
        No data for {yKey}
      </div>
    )
  }

  const xMin = Math.min(...data.map(d => d.x))
  const xMax = Math.max(...data.map(d => d.x))
  const yMin = Math.min(...data.map(d => d.y))
  const yMax = Math.max(...data.map(d => d.y))

  const chartW = width - pad.left - pad.right
  const chartH = height - pad.top - pad.bottom

  const xRange = xMax - xMin || 1
  const yRange = yMax - yMin || 1

  const scaleX = (x: number) => ((x - xMin) / xRange) * chartW
  const scaleY = (y: number) => chartH - ((y - yMin) / yRange) * chartH

  const pathD = data
    .map((d, i) => `${i === 0 ? 'M' : 'L'} ${scaleX(d.x).toFixed(1)} ${scaleY(d.y).toFixed(1)}`)
    .join(' ')

  const yTicks = 4
  const xTicks = 5

  function handleMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    const svgRect = e.currentTarget.getBoundingClientRect()
    const mouseX = e.clientX - svgRect.left - pad.left
    const mouseY = e.clientY - svgRect.top - pad.top

    if (mouseX < 0 || mouseX > chartW || mouseY < 0 || mouseY > chartH) {
      setTooltip(t => ({ ...t, visible: false }))
      return
    }

    // Find nearest data point by x
    const hoverXVal = xMin + (mouseX / chartW) * xRange
    let nearest = data[0]
    let nearestDist = Infinity
    for (const d of data) {
      const dist = Math.abs(d.x - hoverXVal)
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
        width={width}
        height={height}
        style={{ display: 'block', cursor: 'crosshair' }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <g transform={`translate(${pad.left},${pad.top})`}>
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
                  {Math.round(v)}
                </text>
              </g>
            )
          })}
          {/* Data line */}
          <path d={pathD} fill="none" stroke={color} strokeWidth={1.5} />
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
            {xKey}
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
          <span style={{ color: '#888' }}>{xKey}:</span> {Math.round(tooltip.xVal)}&nbsp;&nbsp;
          <span style={{ color: '#888' }}>{yKey}:</span> {fmtY(tooltip.yVal)}
        </div>
      )}
    </>
  )
}

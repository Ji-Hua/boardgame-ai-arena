/**
 * EvalReplayBoard — self-contained SVG board for rendering Quoridor game states
 * from evaluation replay snapshots.
 *
 * Coordinate system (matches engine / replay_writer.py):
 *   Engine: x = column 0-8 (left→right), y = row 0-8 (bottom→top)
 *   Screen: col = x, screen_row = 8 - y (0 at top)
 *
 * Wall rendering:
 *   H wall head (wx, wy): horizontal wall between engine rows wy and wy+1,
 *     spanning engine cols wx and wx+1.
 *     Screen: sits in the gap between screen rows (7-wy) and (8-wy).
 *   V wall head (wx, wy): vertical wall between engine cols wx and wx+1,
 *     spanning engine rows wy and wy+1.
 *     Screen: sits in the gap between screen cols wx and wx+1.
 */

import React from 'react'
import type { EvalReplayState } from '../types'

// -------------------------------------------------------------------------
// Board geometry constants
// -------------------------------------------------------------------------

const CELL = 44          // cell square side in px
const GAP = 7            // gap between cells (wall slot width/height)
const STEP = CELL + GAP  // distance between cell origins
const BORDER = 12        // outer padding

/** Total SVG dimensions */
export const BOARD_SVG_SIZE = 9 * STEP + BORDER * 2 - GAP   // = 9*51 + 24 - 7 = 476

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

function cellLeft(x: number): number {
  return x * STEP + BORDER
}

function cellTop(y_engine: number): number {
  const screen_row = 8 - y_engine
  return screen_row * STEP + BORDER
}

// -------------------------------------------------------------------------
// Component
// -------------------------------------------------------------------------

interface EvalReplayBoardProps {
  state: EvalReplayState
  /** Highlight the last action. */
  lastActionType?: 'pawn' | 'hwall' | 'vwall'
  lastActionX?: number
  lastActionY?: number
  lastActionPlayer?: 'P1' | 'P2'
  size?: number  // optional override for total SVG size (renders via viewBox scaling)
}

const P1_COLOR = '#4a90e2'
const P2_COLOR = '#e24a4a'
const P1_WALL_COLOR = '#2c6fad'
const P2_WALL_COLOR = '#ad2c2c'
const BOARD_BG = '#f5e6c8'
const CELL_COLOR = '#fdf5e6'
const CELL_GOAL_P1 = '#d4edda'   // light green — P1 goal row (y=8, top)
const CELL_GOAL_P2 = '#fcd4d4'   // light red   — P2 goal row (y=0, bottom)
const GRID_COLOR = '#c8a96e'

/** Build a CSS colour from a wall array element's owner. */
function wallColor(player: 'P1' | 'P2' | null): string {
  return player === 'P1' ? P1_WALL_COLOR : player === 'P2' ? P2_WALL_COLOR : '#555'
}

export const EvalReplayBoard: React.FC<EvalReplayBoardProps> = ({
  state,
  lastActionType,
  lastActionX,
  lastActionY,
  lastActionPlayer,
  size,
}) => {
  const viewSize = BOARD_SVG_SIZE

  const cells = []
  for (let y = 0; y <= 8; y++) {
    for (let x = 0; x <= 8; x++) {
      const isGoalP1 = y === 8
      const isGoalP2 = y === 0
      let fill = CELL_COLOR
      if (isGoalP1) fill = CELL_GOAL_P1
      if (isGoalP2) fill = CELL_GOAL_P2
      cells.push(
        <rect
          key={`cell-${x}-${y}`}
          x={cellLeft(x)}
          y={cellTop(y)}
          width={CELL}
          height={CELL}
          fill={fill}
          stroke={GRID_COLOR}
          strokeWidth={0.5}
        />
      )
    }
  }

  // Walls — derive owner from position order for colouring (no owner in snapshot)
  // We render all walls in a neutral colour since ownership isn't tracked in snapshots
  const hWallRects = state.h_walls.map(([wx, wy], i) => {
    // H wall: between screen rows (7-wy) and (8-wy), spanning cols wx and wx+1
    const left = cellLeft(wx)
    const top = cellTop(wy) - GAP   // cellTop(wy) is screen row 8-wy; gap above it
    const width = 2 * CELL + GAP
    const height = GAP
    const isLast = lastActionType === 'hwall' && lastActionX === wx && lastActionY === wy
    const color = isLast ? (lastActionPlayer === 'P1' ? P1_WALL_COLOR : P2_WALL_COLOR) : '#888'
    return (
      <rect
        key={`hw-${i}`}
        x={left}
        y={top}
        width={width}
        height={height}
        fill={color}
        rx={2}
      />
    )
  })

  const vWallRects = state.v_walls.map(([wx, wy], i) => {
    // V wall: between cols wx and wx+1, spanning rows wy and wy+1
    const left = cellLeft(wx) + CELL   // to the right of col wx
    const top = cellTop(wy + 1)        // cellTop(wy+1) = screen row (7-wy)
    const width = GAP
    const height = 2 * CELL + GAP
    const isLast = lastActionType === 'vwall' && lastActionX === wx && lastActionY === wy
    const color = isLast ? (lastActionPlayer === 'P1' ? P1_WALL_COLOR : P2_WALL_COLOR) : '#888'
    return (
      <rect
        key={`vw-${i}`}
        x={left}
        y={top}
        width={width}
        height={height}
        fill={color}
        rx={2}
      />
    )
  })

  // Pawns
  const [p1x, p1y] = state.p1
  const [p2x, p2y] = state.p2
  const pawnRadius = CELL * 0.36

  const p1Highlight = lastActionType === 'pawn' && lastActionPlayer === 'P1'
  const p2Highlight = lastActionType === 'pawn' && lastActionPlayer === 'P2'

  const pawns = [
    <circle
      key="pawn-p1"
      cx={cellLeft(p1x) + CELL / 2}
      cy={cellTop(p1y) + CELL / 2}
      r={pawnRadius}
      fill={P1_COLOR}
      stroke={p1Highlight ? '#ffff00' : '#fff'}
      strokeWidth={p1Highlight ? 3 : 2}
    />,
    <text
      key="pawn-p1-label"
      x={cellLeft(p1x) + CELL / 2}
      y={cellTop(p1y) + CELL / 2 + 4}
      textAnchor="middle"
      fontSize={12}
      fontWeight="bold"
      fill="#fff"
      style={{ pointerEvents: 'none', userSelect: 'none' }}
    >1</text>,
    <circle
      key="pawn-p2"
      cx={cellLeft(p2x) + CELL / 2}
      cy={cellTop(p2y) + CELL / 2}
      r={pawnRadius}
      fill={P2_COLOR}
      stroke={p2Highlight ? '#ffff00' : '#fff'}
      strokeWidth={p2Highlight ? 3 : 2}
    />,
    <text
      key="pawn-p2-label"
      x={cellLeft(p2x) + CELL / 2}
      y={cellTop(p2y) + CELL / 2 + 4}
      textAnchor="middle"
      fontSize={12}
      fontWeight="bold"
      fill="#fff"
      style={{ pointerEvents: 'none', userSelect: 'none' }}
    >2</text>,
  ]

  // Coordinate labels (col letters at bottom, row numbers at left)
  const colLabels = Array.from({ length: 9 }, (_, x) => (
    <text
      key={`col-${x}`}
      x={cellLeft(x) + CELL / 2}
      y={BOARD_SVG_SIZE - 1}
      textAnchor="middle"
      fontSize={9}
      fill="#8a7050"
    >{x}</text>
  ))
  const rowLabels = Array.from({ length: 9 }, (_, y) => (
    <text
      key={`row-${y}`}
      x={4}
      y={cellTop(y) + CELL / 2 + 3}
      textAnchor="middle"
      fontSize={9}
      fill="#8a7050"
    >{y}</text>
  ))

  const svgProps: React.SVGProps<SVGSVGElement> = size
    ? { width: size, height: size, viewBox: `0 0 ${viewSize} ${viewSize}` }
    : { width: viewSize, height: viewSize }

  return (
    <svg
      {...svgProps}
      style={{ background: BOARD_BG, borderRadius: 8, border: '2px solid #c8a96e' }}
    >
      {/* Background */}
      <rect x={0} y={0} width={viewSize} height={viewSize} fill={BOARD_BG} />
      {cells}
      {hWallRects}
      {vWallRects}
      {pawns}
      {colLabels}
      {rowLabels}
    </svg>
  )
}

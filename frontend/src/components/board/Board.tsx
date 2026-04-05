/**
 * Quoridor Board Component
 * 
 * COORDINATE SYSTEM (Engine Convention — Bottom-Left Origin):
 * ===========================================================
 * 
 * RenderState uses engine coordinates: row = engine_y (0 = bottom), col = engine_x (0 = left).
 * This component flips Y for screen rendering: screen top = engine row (boardSize-1).
 * 
 * Flip helpers:
 *   flipPawnRow(r) = boardSize - 1 - r   (pawn rows 0..8)
 *   flipWallRow(r) = boardSize - 2 - r   (wall rows 0..7)
 * 
 * THREE COORDINATE GRIDS:
 * 
 * 1. Pawn Grid (9×9 Logical): P(r, c) where r, c ∈ [0, 8]
 * 2. Wall Grids (8×8 Logical): V(r, c) and H(r, c) where r, c ∈ [0, 7]
 * 3. Expanded Grid (19×19): E(r, c) where r, c ∈ [0, 18]
 * 
 * Reference: /home/jihua/quoridor-core/coordination.md
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import type { PawnData } from './Pawn';
import type { WallData } from './Wall';
import { GridOverlay } from './GridOverlay';
import { HighlightLayer } from './HighlightLayer';
import { getWallPixelGeometry } from '../../utils/wall_geometry';
import { wallsCross, type LogicalWall } from '../../utils/coordinates';
import { PLAYER_COLORS, ERROR_COLOR } from '../../theme/playerColors';
import './Board.css';

export type BoardMode = 'play' | 'replay';

export interface WallPlacement {
  isActive: boolean;
  playerId: 'P1' | 'P2' | null;
}

export interface WallPreview {
  orientation: 'Horizontal' | 'Vertical';
  logicalRow: number;
  logicalCol: number;
  playerId: 'P1' | 'P2';
  isLegal: boolean;
  reason?: string;
}

export interface BoardProps {
  boardSize?: number;
  cellSize?: number;
  pawns: PawnData[];
  walls: WallData[];
  mode: BoardMode;
  legalMoves?: Array<[number, number]>;
  currentPlayer?: 'P1' | 'P2';
  wallPlacement?: WallPlacement;
  wallPreview?: WallPreview | null;
  onUpdateWallPreview?: (preview: WallPreview | null) => void;
  onConfirmWall?: (preview: WallPreview) => void;
  onMovePawn?: (row: number, col: number) => void;
  onPawnClick?: () => void;
  onCancelWallPlacement?: () => void;
  showDebug?: boolean;

  // External wall error from parent component
  externalWallError?: string | null;
  onClearExternalWallError?: () => void;
}

type WallCandidate = {
  orientation: 'Horizontal' | 'Vertical' | null;
  logicalRow: number;
  logicalCol: number;
};

export const Board: React.FC<BoardProps> = ({
  boardSize = 9,
  cellSize = 48,
  pawns,
  walls,
  mode,
  legalMoves = [],
  currentPlayer = 'P1',
  wallPlacement,
  wallPreview,
  onUpdateWallPreview,
  onConfirmWall,
  onMovePawn,
  onPawnClick,
  onCancelWallPlacement,
  showDebug = false,

  externalWallError,
  onClearExternalWallError,
}) => {
  const boardRef = useRef<SVGSVGElement>(null);
  const [wallError, setWallError] = useState<string | null>(null);

  // ⭐ 接收来自 PlayGame.tsx 的 error，显示在棋盘上方
  useEffect(() => {
    if (externalWallError) {
      setWallError(externalWallError);
      onClearExternalWallError?.();
    }
  }, [externalWallError, onClearExternalWallError]);

  // 自动隐藏错误
  useEffect(() => {
    if (wallError) {
      const t = setTimeout(() => setWallError(null), 2000);
      return () => clearTimeout(t);
    }
  }, [wallError]);

  // ESC 取消放墙
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (wallPlacement?.isActive && e.key === 'Escape') {
        onCancelWallPlacement?.();
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [wallPlacement?.isActive, onCancelWallPlacement]);

  // 检查棋子移动合法
  const isLegalMove = useCallback(
    (row: number, col: number) => legalMoves.some(([r, c]) => r === row && c === col),
    [legalMoves]
  );

  // ---------------------------------------------------------------------------
  // WALL EDGE DETECTION
  // ---------------------------------------------------------------------------
  const getWallFromMousePosition = useCallback(
    (x: number, y: number, rect: DOMRect): WallCandidate => {
      const cw = rect.width / boardSize;
      const ch = rect.height / boardSize;

      const rowFloat = (y - rect.top) / ch;
      const colFloat = (x - rect.left) / cw;

      const fracRow = rowFloat - Math.floor(rowFloat);
      const fracCol = colFloat - Math.floor(colFloat);

      const distTop = fracRow;
      const distBottom = 1 - fracRow;
      const distLeft = fracCol;
      const distRight = 1 - fracCol;

      const minV = Math.min(distTop, distBottom);
      const minH = Math.min(distLeft, distRight);

      let orientation: 'Horizontal' | 'Vertical' | null;
      let logicalRow: number;
      let logicalCol: number;

      if (minV < minH) {
        orientation = 'Horizontal';
        logicalRow = fracRow < 0.5 ? Math.floor(rowFloat) - 1 : Math.floor(rowFloat);
        logicalCol = Math.floor(colFloat);
      } else {
        orientation = 'Vertical';
        logicalRow = Math.floor(rowFloat);
        logicalCol = fracCol < 0.5 ? Math.floor(colFloat) - 1 : Math.floor(colFloat);
      }

      if (
        logicalRow < 0 ||
        logicalRow > boardSize - 2 ||
        logicalCol < 0 ||
        logicalCol > boardSize - 2
      ) {
        return { orientation: null, logicalRow: 0, logicalCol: 0 };
      }

      return { orientation, logicalRow, logicalCol };
    },
    [boardSize]
  );

  // ---------------------------------------------------------------------------
  // WALL VALIDATION (overlap, crossing - GOLDEN STANDARD)
  // ---------------------------------------------------------------------------
  // Frontend performs basic validation. Backend does authoritative BFS path check.
  // 
  // ILLEGAL placements:
  //   - Exact duplicate: same orientation and logical position
  //   - Crossing: different orientations at same (r,c) share midpoint
  //   - True overlap: same orientation, adjacent positions (distance 1)
  // 
  // LEGAL placements:
  //   - Connected: walls touch at vertices only (distance 2, same orientation)
  //   - Separated: walls don't touch
  // 
  // NOTE: Path-blocking (BFS) is NOT checked on frontend - backend authoritative.
  // ---------------------------------------------------------------------------
  const canPlaceWall = useCallback(
    (
      orientation: 'Horizontal' | 'Vertical',
      row: number,
      col: number
    ): { isLegal: boolean; reason?: string } => {
      const short: 'H' | 'V' = orientation === 'Horizontal' ? 'H' : 'V';
      const newWall: LogicalWall = { orientation: short, row, col };

      // Rule 1: Check for exact duplicate (same position and orientation)
      const hasDuplicate = walls.some(
        (w) => w.orientation === short && w.row === row && w.col === col
      );
      if (hasDuplicate) {
        return { isLegal: false, reason: 'Wall already exists at this position' };
      }

      // Rule 2: Check for crossing using wallsCross utility
      // Walls cross if they have different orientations at same (r,c)
      const hasCrossing = walls.some((w) => {
        const existingWall: LogicalWall = { orientation: w.orientation as 'H' | 'V', row: w.row, col: w.col };
        return wallsCross(newWall, existingWall);
      });
      if (hasCrossing) {
        return { isLegal: false, reason: 'Walls cannot cross each other' };
      }

      // Rule 3: Check for true overlap (same orientation, adjacent positions)
      // True overlap means walls share 2+ expanded cells (distance 1)
      // Connected walls at distance 2 share only 1 vertex cell - LEGAL
      const hasTrueOverlap = walls.some((w) => {
        if (w.orientation !== short) return false;
        
        if (short === 'H') {
          // Horizontal: overlap if same row and columns differ by 1 or 0
          return w.row === row && Math.abs(w.col - col) <= 1;
        } else {
          // Vertical: overlap if same col and rows differ by 1 or 0
          return w.col === col && Math.abs(w.row - row) <= 1;
        }
      });
      if (hasTrueOverlap) {
        return { isLegal: false, reason: 'Walls cannot overlap' };
      }

      return { isLegal: true };
    },
    [walls]
  );

  // ---------------------------------------------------------------------------
  // PREVIEW HANDLER
  // ---------------------------------------------------------------------------
  const handleMouseMove = useCallback(
    (event: React.MouseEvent) => {
      if (!wallPlacement?.isActive || !wallPlacement.playerId) return;

      const rect = boardRef.current?.getBoundingClientRect();
      if (!rect) return;

      const c = getWallFromMousePosition(event.clientX, event.clientY, rect);
      if (c.orientation === null) {
        onUpdateWallPreview?.(null);
        return;
      }

      // Convert screen wall row to engine wall row (flip Y)
      const engineRow = boardSize - 2 - c.logicalRow;
      const validation = canPlaceWall(c.orientation, engineRow, c.logicalCol);

      onUpdateWallPreview?.({
        orientation: c.orientation,
        logicalRow: engineRow,
        logicalCol: c.logicalCol,
        playerId: wallPlacement.playerId,
        isLegal: validation.isLegal,
        reason: validation.reason,
      });
    },
    [wallPlacement, getWallFromMousePosition, canPlaceWall, onUpdateWallPreview]
  );

  // ---------------------------------------------------------------------------
  // CLICK: CONFIRM WALL
  // ---------------------------------------------------------------------------
  const handleBoardClick = useCallback(
    (event: React.MouseEvent) => {
      if (event.button !== 0) return;

      if (wallPlacement?.isActive && wallPreview) {
        if (!wallPreview.isLegal) {
          setWallError(wallPreview.reason ?? 'Invalid wall position');
          return;
        }

        onConfirmWall?.(wallPreview);
        return;
      }
    },
    [wallPlacement, wallPreview, onConfirmWall]
  );

  // ---------------------------------------------------------------------------
  // MISC
  // ---------------------------------------------------------------------------
  const handleContextMenu = useCallback(
    (event: React.MouseEvent) => {
      if (wallPlacement?.isActive) {
        event.preventDefault();
        onCancelWallPlacement?.();
      }
    },
    [wallPlacement, onCancelWallPlacement]
  );

  const handleMouseLeave = useCallback(() => {
    if (wallPlacement?.isActive) {
      onUpdateWallPreview?.(null);
    }
  }, [wallPlacement, onUpdateWallPreview]);

  const handleCellClick = useCallback(
    (row: number, col: number) => {
      if (wallPlacement?.isActive) return;
      if (mode === 'play') {
        // If click is on the current player's pawn, trigger pawn-click (legal moves fetch)
        const currentPawn = pawns.find((p) => p.playerId === currentPlayer);
        if (onPawnClick && currentPawn && currentPawn.row === row && currentPawn.col === col) {
          onPawnClick();
          return;
        }
        if (onMovePawn) {
          // Backend-authoritative: send any click, backend validates
          onMovePawn(row, col);
        }
      }
    },
    [wallPlacement, mode, onMovePawn, onPawnClick, pawns, currentPlayer]
  );

  const gridWidth = boardSize * cellSize;
  const gridHeight = boardSize * cellSize;

  // Calculate board offset (padding + border)
  const boardOffsetX = 24 + 2; // board-wrapper padding (24px) + SVG border (2px)
  const boardOffsetY = 24 + 2; // board-wrapper padding (24px) + SVG border (2px)

  // Coordinate to pixel mapping function (engine coords → screen pixels)
  // Flip Y: engine row 0 = screen bottom, engine row (boardSize-1) = screen top
  const toPixel = useCallback((row: number, col: number) => {
    return {
      x: boardOffsetX + col * cellSize + cellSize / 2,
      y: boardOffsetY + (boardSize - 1 - row) * cellSize + cellSize / 2,
    };
  }, [cellSize, boardOffsetX, boardOffsetY, boardSize]);

  // Dev-only console test for pixel coordinates
  useEffect(() => {
    if (showDebug) {
      console.log('Board pixel coordinates test:');
      console.log('Corner (0,0):', toPixel(0, 0));
      console.log('Corner (0,8):', toPixel(0, 8));
      console.log('Corner (8,0):', toPixel(8, 0));
      console.log('Corner (8,8):', toPixel(8, 8));
    }
  }, [toPixel, showDebug]);

  // ---------------------------------------------------------------------------
  // RENDER
  // ---------------------------------------------------------------------------
  return (
    <div className="board-wrapper" style={{ position: 'relative' }}>
      {wallError && <div className="wall-error">{wallError}</div>}

      {/* Highlight Layer for legal pawn moves */}
      {mode === 'play' && legalMoves && legalMoves.length > 0 && onMovePawn && (
        <HighlightLayer
          legalPawnMoves={legalMoves.map(([row, col]) => ({ row, col }))}
          currentPlayerColor={PLAYER_COLORS[currentPlayer || 'P1'].primary}
          cellSize={cellSize}
          toPixel={toPixel}
          onPawnMove={onMovePawn}
        />
      )}

      <svg
        ref={boardRef}
        width={gridWidth}
        height={gridHeight}
        className="board-container"
        overflow="visible"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        onContextMenu={handleContextMenu}
        onClick={handleBoardClick}
        style={{ cursor: wallPlacement?.isActive ? 'crosshair' : 'default' }}
      >
        <rect width={gridWidth} height={gridHeight} fill="#ffffff" />

        {/* Grid */}
        <GridOverlay boardSize={boardSize} cellSize={cellSize} showDebug={showDebug} />

        {/* Wall preview (convert engine row to screen row for rendering) */}
        {wallPreview && (
          <WallPreviewSVG
            orientation={wallPreview.orientation}
            logicalRow={boardSize - 2 - wallPreview.logicalRow}
            logicalCol={wallPreview.logicalCol}
            cellSize={cellSize}
            color={
              wallPreview.isLegal
                ? PLAYER_COLORS[wallPreview.playerId].primary
                : ERROR_COLOR
            }
            isLegal={wallPreview.isLegal}
          />
        )}

        {/* Existing walls (convert engine row to screen row for rendering) */}
        {walls.map((wall, index) => (
          <WallSVG
            key={`wall-${index}`}
            orientation={wall.orientation === 'H' ? 'Horizontal' : 'Vertical'}
            logicalRow={boardSize - 2 - wall.row}
            logicalCol={wall.col}
            cellSize={cellSize}
            color={wall.color}
          />
        ))}

        {/* Pawns (flip Y: engine row 0 = screen bottom) */}
        {pawns.map((pawn) => {
          const cx = (pawn.col + 0.5) * cellSize;
          const cy = (boardSize - 1 - pawn.row + 0.5) * cellSize;
          const r = cellSize * 0.35;
          return (
            <g key={`pawn-${pawn.playerId}`}>
              <circle cx={cx} cy={cy} r={r} fill={pawn.color} />
              <text
                x={cx}
                y={cy}
                textAnchor="middle"
                dominantBaseline="middle"
                fontSize={14}
                fontWeight="bold"
                fill="white"
                pointerEvents="none"
              >
                {pawn.playerId}
              </text>
            </g>
          );
        })}

        {/* Invisible click targets (convert screen row to engine row) */}
        {mode === 'play' &&
          !wallPlacement?.isActive &&
          Array.from({ length: boardSize }, (_, r) =>
            Array.from({ length: boardSize }, (_, c) => (
              <rect
                key={`cell-${r}-${c}`}
                x={c * cellSize}
                y={r * cellSize}
                width={cellSize}
                height={cellSize}
                fill="transparent"
                onClick={() => handleCellClick(boardSize - 1 - r, c)}
              />
            ))
          )}
      </svg>
    </div>
  );
};

// ============================================================================
// WALL SVG RENDERERS - GOLDEN STANDARD (19×19 Expanded Grid)
// ============================================================================
// 
// Each wall occupies 5 cells in the expanded 19×19 grid:
//   vertex → edge → midpoint → edge → vertex
//
// Horizontal Wall H(r, c):
//   [E(2r+2, 2c), E(2r+2, 2c+1), E(2r+2, 2c+2), E(2r+2, 2c+3), E(2r+2, 2c+4)]
//
// Vertical Wall V(r, c):
//   [E(2r, 2c+2), E(2r+1, 2c+2), E(2r+2, 2c+2), E(2r+3, 2c+2), E(2r+4, 2c+2)]
//
// Rendering approach:
//   - Logical 9×9 grid cells map to display cells
//   - Each logical cell size = cellSize pixels
//   - Expanded grid has 19×19 cells (including walls)
//   - Display cell spacing = cellSize / 2 for sub-cell precision
// ============================================================================

// WALL_THICKNESS is now sourced from utils/wall_geometry

const WallPreviewSVG: React.FC<{
  orientation: 'Horizontal' | 'Vertical';
  logicalRow: number;
  logicalCol: number;
  cellSize: number;
  color: string;
  isLegal: boolean;
}> = ({ orientation, logicalRow, logicalCol, cellSize, color, isLegal }) => {
  // Create a lighter version of the color for preview
  const getPreviewColor = (baseColor: string) => {
    return baseColor.replace(/rgba?\([^)]+\)/, (match) => {
      const rgba = match.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)/);
      if (rgba) {
        const r = parseInt(rgba[1]);
        const g = parseInt(rgba[2]);
        const b = parseInt(rgba[3]);
        return `rgba(${r}, ${g}, ${b}, 0.35)`;
      }
      return baseColor;
    });
  };

  const previewColor = isLegal ? getPreviewColor(color) : 'rgba(220, 53, 69, 0.8)';
  const geom = getWallPixelGeometry(orientation, logicalRow, logicalCol, cellSize);

  return (
    <rect
      x={geom.x}
      y={geom.y}
      width={geom.width}
      height={geom.height}
      fill={previewColor}
      stroke={previewColor}
      strokeWidth={1}
      opacity={0.7}
      rx={3}
      style={{ filter: `drop-shadow(0 0 6px ${previewColor})` }}
    />
  );
};

const WallSVG: React.FC<{
  orientation: 'Horizontal' | 'Vertical';
  logicalRow: number;
  logicalCol: number;
  cellSize: number;
  color: string;
}> = ({ orientation, logicalRow, logicalCol, cellSize, color }) => {
  // Create a darker version of the wall color for placed walls
  const getWallColor = (baseColor: string) => {
    return baseColor.replace(/rgba?\([^)]+\)/, (match) => {
      const rgba = match.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)/);
      if (rgba) {
        const r = parseInt(rgba[1]);
        const g = parseInt(rgba[2]);
        const b = parseInt(rgba[3]);
        return `rgba(${Math.max(0, r - 40)}, ${Math.max(0, g - 40)}, ${Math.max(0, b - 40)}, 0.95)`;
      }
      return baseColor;
    });
  };

  const wallColor = getWallColor(color);
  const geom = getWallPixelGeometry(orientation, logicalRow, logicalCol, cellSize);
  return (
    <rect
      x={geom.x}
      y={geom.y}
      width={geom.width}
      height={geom.height}
      fill={wallColor}
      stroke={wallColor}
      strokeWidth={2}
      rx={3}
      style={{ filter: `drop-shadow(0 0 4px ${wallColor})` }}
    />
  );
};

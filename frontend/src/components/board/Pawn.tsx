import React from 'react';

export interface PawnData {
  playerId: 'P1' | 'P2';
  row: number;
  col: number;
  color: string;
}

interface PawnProps {
  pawn: PawnData;
  cellSize: number;
  size?: number;
  isHighlighted?: boolean;
  onClick?: () => void;
}

/**
 * Pawn component for rendering player pieces on the board
 * Renders as a circle centered exactly in the cell
 */
export const Pawn: React.FC<PawnProps> = ({
  pawn,
  cellSize,
  size,
  isHighlighted = false,
  onClick,
}) => {
  // Calculate size as proportion of cell (radius ≈ 0.35 * cellSize)
  const pawnSize = size || Math.floor(cellSize * 0.7);
  
  // Center the pawn in its cell: (col + 0.5) * cellSize, (row + 0.5) * cellSize
  const centerX = (pawn.col + 0.5) * cellSize;
  const centerY = (pawn.row + 0.5) * cellSize;
  const left = centerX - pawnSize / 2;
  const top = centerY - pawnSize / 2;

  const pawnStyle: React.CSSProperties = {
    left: `${left}px`,
    top: `${top}px`,
    width: `${size}px`,
    height: `${size}px`,
    borderRadius: '50%',
    backgroundColor: pawn.color,
    border: isHighlighted ? '3px solid #ffd700' : '2px solid #333',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    cursor: onClick ? 'pointer' : 'default',
    transition: 'all 0.3s ease',
    boxShadow: isHighlighted ? '0 0 10px rgba(255, 215, 0, 0.5)' : '0 2px 4px rgba(0,0,0,0.2)',
    position: 'absolute',
    zIndex: 10,
  };

  const labelStyle: React.CSSProperties = {
    color: 'white',
    fontSize: '14px',
    fontWeight: 'bold',
    textShadow: '1px 1px 2px rgba(0,0,0,0.7)',
    userSelect: 'none',
  };

  return (
    <div
      style={pawnStyle}
      onClick={onClick}
      title={`Player ${pawn.playerId}`}
    >
      <span style={labelStyle}>{pawn.playerId}</span>
    </div>
  );
};
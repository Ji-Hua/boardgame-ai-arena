import React from 'react';

interface GridOverlayProps {
  boardSize: number;
  cellSize: number;
  showDebug?: boolean;
}

/**
 * Grid overlay component that renders light grey grid lines
 * with numeric axis labels matching engine coordinate system.
 * X axis (columns 0-8) at bottom, Y axis (rows 0-8) on left (0=bottom, 8=top).
 */
export const GridOverlay: React.FC<GridOverlayProps> = ({
  boardSize,
  cellSize,
}) => {
  const gridLines: React.ReactElement[] = [];
  const labels: React.ReactElement[] = [];

  // Horizontal grid lines
  for (let row = 0; row <= boardSize; row++) {
    gridLines.push(
      <line
        key={`h-${row}`}
        x1="0"
        y1={row * cellSize}
        x2={boardSize * cellSize}
        y2={row * cellSize}
        stroke="#d0d0d0"
        strokeWidth="1"
      />
    );
  }

  // Vertical grid lines
  for (let col = 0; col <= boardSize; col++) {
    gridLines.push(
      <line
        key={`v-${col}`}
        x1={col * cellSize}
        y1="0"
        x2={col * cellSize}
        y2={boardSize * cellSize}
        stroke="#d0d0d0"
        strokeWidth="1"
      />
    );
  }

  // X axis labels (engine x = 0..8) at bottom of board
  for (let col = 0; col < boardSize; col++) {
    labels.push(
      <text
        key={`x-${col}`}
        x={col * cellSize + cellSize / 2}
        y={boardSize * cellSize + cellSize * 0.3}
        textAnchor="middle"
        fontSize={cellSize * 0.25}
        fill="#666"
        fontWeight="bold"
      >
        {col}
      </text>
    );
  }

  // Y axis labels (engine y = 0..8) on left of board
  // Screen row 0 (top) = engine y (boardSize-1), screen row 8 (bottom) = engine y 0
  for (let screenRow = 0; screenRow < boardSize; screenRow++) {
    const engineY = boardSize - 1 - screenRow;
    labels.push(
      <text
        key={`y-${screenRow}`}
        x={-cellSize * 0.15}
        y={screenRow * cellSize + cellSize / 2}
        textAnchor="end"
        dominantBaseline="middle"
        fontSize={cellSize * 0.25}
        fill="#666"
        fontWeight="bold"
      >
        {engineY}
      </text>
    );
  }

  return (
    <>
      {gridLines}
      {labels}
    </>
  );
};
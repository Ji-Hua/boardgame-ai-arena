/**
 * HighlightLayer Component
 * 
 * Renders legal move highlights on the board using legal_actions from backend.
 * Shows colored circles at legal pawn positions for the current player.
 * Circles are interactive and clickable to move pawns.
 */

import React from 'react';
import './HighlightLayer.css';

interface HighlightLayerProps {
  legalPawnMoves: Array<{ row: number; col: number }>;
  currentPlayerColor: string;
  cellSize: number;
  toPixel: (row: number, col: number) => { x: number; y: number };
  onPawnMove: (row: number, col: number) => void;
}

export const HighlightLayer: React.FC<HighlightLayerProps> = ({
  legalPawnMoves,
  currentPlayerColor,
  cellSize,
  toPixel,
  onPawnMove,
}) => {
  if (!legalPawnMoves || legalPawnMoves.length === 0) {
    return null;
  }

  return (
    <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 10 }}>
      {legalPawnMoves.map((move, index) => {
        const { x, y } = toPixel(move.row, move.col);
        const radius = cellSize * 0.22;
        
        return (
          <div
            key={`legal-move-${index}`}
            className="highlight-circle"
            onClick={() => onPawnMove(move.row, move.col)}
            style={{
              position: 'absolute',
              left: x - radius,
              top: y - radius,
              width: radius * 2,
              height: radius * 2,
              borderRadius: '50%',
              backgroundColor: currentPlayerColor,
              opacity: 0.4,
              border: `2px solid ${currentPlayerColor}`,
              boxShadow: `0 0 8px ${currentPlayerColor}`,
              pointerEvents: 'auto',
              cursor: 'pointer',
            }}
          />
        );
      })}
    </div>
  );
};

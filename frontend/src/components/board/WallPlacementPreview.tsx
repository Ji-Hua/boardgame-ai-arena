/**
 * Wall Placement Preview Component
 * 
 * COORDINATE SYSTEM (GOLDEN STANDARD - Matrix Style):
 * ==================================================
 * 
 * Renders wall preview using 5-segment expanded 19×19 grid geometry.
 * See /home/jihua/quoridor-core/coordination.md for full specification.
 * 
 * Horizontal Wall H(r, c): 5 cells from E(2r+2, 2c) to E(2r+2, 2c+4)
 * Vertical Wall V(r, c): 5 cells from E(2r, 2c+2) to E(2r+4, 2c+2)
 */

import React from 'react';

interface WallPlacementPreviewProps {
  orientation: 'H' | 'V';
  row: number;  // Logical wall grid (0-7)
  col: number;  // Logical wall grid (0-7)
  color: string;
  isValid: boolean;
  cellSize?: number;  // Logical cell size (default 50)
}

/**
 * Preview component for wall placement during hover/selection
 * Uses GOLDEN STANDARD 5-segment expanded grid geometry
 */
export const WallPlacementPreview: React.FC<WallPlacementPreviewProps> = ({
  orientation,
  row,
  col,
  color,
  isValid,
  cellSize = 50,
}) => {
  const wallThickness = 8; // Increased thickness (was 6)
  const expandedCellSize = cellSize / 2;
  const wallLength = 5 * expandedCellSize + wallThickness;

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

  const previewColor = isValid ? getPreviewColor(color) : 'rgba(220, 53, 69, 0.8)';

  const previewStyle: React.CSSProperties = {
    position: 'absolute',
    backgroundColor: previewColor,
    border: `1px solid ${previewColor}`,
    borderRadius: '3px',
    opacity: 0.7,
    zIndex: 3,
    pointerEvents: 'none',
    transition: 'all 0.2s ease',
    boxShadow: `0 0 6px ${previewColor}`,
  };

  if (orientation === 'H') {
    // Horizontal wall H(r, c) in expanded grid
    const expRow = 2 * row + 2;
    const expColStart = 2 * col;
    
    previewStyle.width = `${wallLength}px`;
    previewStyle.height = `${wallThickness}px`;
    previewStyle.left = `${expColStart * expandedCellSize - wallThickness / 2}px`;
    previewStyle.top = `${expRow * expandedCellSize - wallThickness / 2}px`;
  } else {
    // Vertical wall V(r, c) in expanded grid
    const expCol = 2 * col + 2;
    const expRowStart = 2 * row;
    
    previewStyle.width = `${wallThickness}px`;
    previewStyle.height = `${wallLength}px`;
    previewStyle.left = `${expCol * expandedCellSize - wallThickness / 2}px`;
    previewStyle.top = `${expRowStart * expandedCellSize - wallThickness / 2}px`;
  }

  return <div style={previewStyle} />;
};
/**
 * Wall Component
 * 
 * COORDINATE SYSTEM (GOLDEN STANDARD - Matrix Style):
 * ==================================================
 * 
 * Walls use logical coordinates (8×8 grid) and are rendered using
 * expanded 19×19 grid geometry (5 segments per wall).
 * 
 * Horizontal Wall H(r, c):
 *   Logical position: r, c ∈ [0, 7]
 *   Expanded cells: [E(2r+2, 2c), E(2r+2, 2c+1), E(2r+2, 2c+2), E(2r+2, 2c+3), E(2r+2, 2c+4)]
 *   Row fixed at 2r+2 (even), columns vary from 2c to 2c+4
 *   Pattern: vertex → edge → midpoint → edge → vertex
 * 
 * Vertical Wall V(r, c):
 *   Logical position: r, c ∈ [0, 7]
 *   Expanded cells: [E(2r, 2c+2), E(2r+1, 2c+2), E(2r+2, 2c+2), E(2r+3, 2c+2), E(2r+4, 2c+2)]
 *   Column fixed at 2c+2 (even), rows vary from 2r to 2r+4
 *   Pattern: vertex → edge → midpoint → edge → vertex
 * 
 * RENDERING:
 * - Each expanded cell = cellSize/2 pixels
 * - Total wall length = 5 * (cellSize/2) = 2.5 * cellSize pixels
 * - This replaces the old 2-cell wall rendering
 * 
 * Reference: /home/jihua/quoridor-core/coordination.md
 */

import React from 'react';
import { getWallPixelGeometry } from '../../utils/wall_geometry';

export interface WallData {
  orientation: 'H' | 'V';
  row: number;  // Logical wall grid (0-7)
  col: number;  // Logical wall grid (0-7)
  owner?: 'P1' | 'P2';
  color: string;
}

interface WallProps {
  wall: WallData;
  cellSize: number;
  isPreview?: boolean;
  isInvalid?: boolean;
  onClick?: () => void;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;
}

/**
 * Wall component for rendering placed walls on the board
 * Uses GOLDEN STANDARD 5-segment expanded grid geometry
 */
export const Wall: React.FC<WallProps> = ({
  wall,
  cellSize,
  isPreview = false,
  isInvalid = false,
  onClick,
  onMouseEnter,
  onMouseLeave,
}) => {
  // Create a darker version of the wall color for placed walls
  const getWallColor = (baseColor: string) => {
    if (isPreview) {
      // Lighter version for preview walls
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
    } else {
      // Darker, more opaque version for placed walls
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
    }
  };

  const wallColor = getWallColor(wall.color);

  const wallStyle: React.CSSProperties = {
    position: 'absolute',
    backgroundColor: wallColor,
    border: isPreview ? 'none' : `2px solid ${wallColor}`,
    borderRadius: '3px',
    transition: 'all 0.3s ease',
    opacity: isPreview ? 0.7 : 1,
    cursor: onClick ? 'pointer' : 'default',
    zIndex: isPreview ? 5 : 2,
    boxShadow: isPreview ? 'none' : `0 0 4px ${wallColor}`,
  };

  const geom = getWallPixelGeometry(wall.orientation === 'H' ? 'H' : 'V', wall.row, wall.col, cellSize);
  wallStyle.width = `${geom.width}px`;
  wallStyle.height = `${geom.height}px`;
  wallStyle.left = `${geom.x}px`;
  wallStyle.top = `${geom.y}px`;

  return (
    <div
      style={wallStyle}
      onClick={onClick}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      title={wall.owner ? `Wall placed by ${wall.owner}` : 'Wall'}
    />
  );
};
/**
 * Wall geometry utilities for frontend rendering
 *
 * Provides pixel geometry for walls such that a placed wall visually spans
 * exactly two logical pawn cells on the 9x9 board.
 */
export const WALL_THICKNESS = 6;

type Orientation = 'H' | 'V' | 'Horizontal' | 'Vertical';

export function getWallPixelGeometry(
  orientation: Orientation,
  logicalRow: number,
  logicalCol: number,
  cellSize: number
): { x: number; y: number; width: number; height: number } {
  const isHorizontal = orientation === 'H' || orientation === 'Horizontal';
  const thickness = WALL_THICKNESS;

  if (isHorizontal) {
    // Horizontal wall spans two logical columns: from col -> col+1
    const width = 2 * cellSize;
    const height = thickness;

    // x starts at left edge of logicalCol cell
    const x = logicalCol * cellSize;

    // centered between logicalRow and logicalRow+1: centerY = (logicalRow+1)*cellSize
    // top y is centerY - height/2
    const y = (logicalRow + 1) * cellSize - height / 2;

    return { x, y, width, height };
  }

  // Vertical wall spans two logical rows: from row -> row+1
  const width = thickness;
  const height = 2 * cellSize;

  // centered between logicalCol and logicalCol+1: centerX = (logicalCol+1)*cellSize
  // left x is centerX - width/2
  const x = (logicalCol + 1) * cellSize - width / 2;
  const y = logicalRow * cellSize;

  return { x, y, width, height };
}

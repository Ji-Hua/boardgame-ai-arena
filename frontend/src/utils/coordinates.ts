/**
 * Coordinate System Utilities for Quoridor
 * 
 * GOLDEN STANDARD - Matrix Style Coordinate System
 * ================================================
 * 
 * This file implements the authoritative coordinate system defined in:
 * quoridor-core/coordination.md
 * 
 * THREE COORDINATE GRIDS:
 * 
 * 1. Pawn Grid (9×9 Logical): P(r, c) where r, c ∈ [0, 8]
 *    - Pawn positions in logical coordinate system
 *    - Row 0 = top of board, row 8 = bottom of board
 *    - Note: This is the INTERNAL logical grid used for rendering calculations.
 *      Display Grid Protocol (A-I, 1-9) is converted to/from this via displayGridConverter.ts
 *    - Display Grid: P1 starts at E1 (rank 1) = logical row 8, P2 starts at E9 (rank 9) = logical row 0
 * 
 * 2. Wall Grids (8×8 Logical): V(r, c) and H(r, c) where r, c ∈ [0, 7]
 *    - Vertical walls: V(r, c)
 *    - Horizontal walls: H(r, c)
 * 
 * 3. Expanded Grid (19×19): E(r, c) where r, c ∈ [0, 18]
 *    - Unified representation for pawns and walls
 *    - Pawn cells occupy (odd, odd) coordinates
 *    - Wall segments occupy mixed coordinates
 * 
 * COORDINATE MAPPINGS:
 * 
 * Pawn: P(r, c) → E(2r+1, 2c+1)
 * 
 * Vertical Wall V(r, c) occupies 5 cells:
 *   [E(2r, 2c+2), E(2r+1, 2c+2), E(2r+2, 2c+2), E(2r+3, 2c+2), E(2r+4, 2c+2)]
 *   - Column fixed at 2c+2 (even: 2, 4, 6, ..., 16)
 *   - Rows vary from 2r to 2r+4
 * 
 * Horizontal Wall H(r, c) occupies 5 cells:
 *   [E(2r+2, 2c), E(2r+2, 2c+1), E(2r+2, 2c+2), E(2r+2, 2c+3), E(2r+2, 2c+4)]
 *   - Row fixed at 2r+2 (even: 2, 4, 6, ..., 16)
 *   - Columns vary from 2c to 2c+4
 * 
 * WALL STRUCTURE (5 segments):
 *   vertex → edge → midpoint → edge → vertex
 * 
 * WALL LEGALITY RULES:
 * - Connected walls (touching at vertex/midpoint): LEGAL
 * - Crossing walls (sharing midpoint): ILLEGAL
 * - Overlapping walls (sharing edge segment): ILLEGAL
 * - Walls cannot run along boundary (r=0, r=18, c=0, c=18)
 * - Both players must have a path to goal (BFS check)
 */

export interface ExpandedCell {
  row: number;
  col: number;
}

export interface LogicalWall {
  orientation: 'H' | 'V';
  row: number;
  col: number;
}

/**
 * Convert logical pawn position P(r, c) to expanded grid E(r', c')
 * 
 * Formula: P(r, c) → E(2r+1, 2c+1)
 * 
 * @param logicalRow - Pawn row in logical grid (0-8)
 * @param logicalCol - Pawn column in logical grid (0-8)
 * @returns Expanded grid coordinates (both odd, range [1, 17])
 * 
 * @example
 * pawnToExpanded(0, 0) // → {row: 1, col: 1}
 * pawnToExpanded(0, 4) // → {row: 1, col: 9}
 * pawnToExpanded(8, 8) // → {row: 17, col: 17}
 */
export function pawnToExpanded(logicalRow: number, logicalCol: number): ExpandedCell {
  return {
    row: 2 * logicalRow + 1,
    col: 2 * logicalCol + 1,
  };
}

/**
 * Convert expanded grid coordinates E(r, c) to logical pawn position P(r', c')
 * 
 * Formula: E(r, c) → P((r-1)/2, (c-1)/2)
 * 
 * @param expandedRow - Row in expanded grid (0-18, should be odd for pawns)
 * @param expandedCol - Column in expanded grid (0-18, should be odd for pawns)
 * @returns Logical pawn coordinates (range [0, 8])
 */
export function expandedToPawn(expandedRow: number, expandedCol: number): ExpandedCell {
  return {
    row: (expandedRow - 1) / 2,
    col: (expandedCol - 1) / 2,
  };
}

/**
 * Convert logical wall position to expanded grid cells (5 segments)
 * 
 * Vertical Wall V(r, c):
 *   [E(2r, 2c+2), E(2r+1, 2c+2), E(2r+2, 2c+2), E(2r+3, 2c+2), E(2r+4, 2c+2)]
 *   Column fixed at 2c+2 (even), rows vary from 2r to 2r+4
 * 
 * Horizontal Wall H(r, c):
 *   [E(2r+2, 2c), E(2r+2, 2c+1), E(2r+2, 2c+2), E(2r+2, 2c+3), E(2r+2, 2c+4)]
 *   Row fixed at 2r+2 (even), columns vary from 2c to 2c+4
 * 
 * @param orientation - "H" for horizontal, "V" for vertical
 * @param logicalRow - Wall row in logical grid (0-7)
 * @param logicalCol - Wall column in logical grid (0-7)
 * @returns Array of 5 expanded grid cells (in order: vertex-edge-midpoint-edge-vertex)
 * 
 * @example
 * // Horizontal wall H(0, 0)
 * wallToExpanded('H', 0, 0) 
 * // → [{row:2,col:0}, {row:2,col:1}, {row:2,col:2}, {row:2,col:3}, {row:2,col:4}]
 * 
 * // Vertical wall V(0, 0)
 * wallToExpanded('V', 0, 0)
 * // → [{row:0,col:2}, {row:1,col:2}, {row:2,col:2}, {row:3,col:2}, {row:4,col:2}]
 */
export function wallToExpanded(
  orientation: 'H' | 'V',
  logicalRow: number,
  logicalCol: number
): ExpandedCell[] {
  if (orientation === 'H') {
    const expRow = 2 * logicalRow + 2;
    const expCol = 2 * logicalCol;
    return [
      { row: expRow, col: expCol },
      { row: expRow, col: expCol + 1 },
      { row: expRow, col: expCol + 2 },
      { row: expRow, col: expCol + 3 },
      { row: expRow, col: expCol + 4 },
    ];
  } else {
    const expRow = 2 * logicalRow;
    const expCol = 2 * logicalCol + 2;
    return [
      { row: expRow, col: expCol },
      { row: expRow + 1, col: expCol },
      { row: expRow + 2, col: expCol },
      { row: expRow + 3, col: expCol },
      { row: expRow + 4, col: expCol },
    ];
  }
}

/**
 * Get the midpoint cell of a wall in expanded grid
 * The midpoint is the 3rd segment (index 2) of the 5-segment wall
 * 
 * @param orientation - "H" for horizontal, "V" for vertical
 * @param logicalRow - Wall row in logical grid (0-7)
 * @param logicalCol - Wall column in logical grid (0-7)
 * @returns Midpoint cell in expanded grid
 * 
 * @example
 * wallMidpoint('H', 0, 0) // → {row: 2, col: 2}
 * wallMidpoint('V', 0, 0) // → {row: 2, col: 2}
 */
export function wallMidpoint(
  orientation: 'H' | 'V',
  logicalRow: number,
  logicalCol: number
): ExpandedCell {
  if (orientation === 'H') {
    return {
      row: 2 * logicalRow + 2,
      col: 2 * logicalCol + 2,
    };
  } else {
    return {
      row: 2 * logicalRow + 2,
      col: 2 * logicalCol + 2,
    };
  }
}

/**
 * Get the vertex endpoints of a wall in expanded grid
 * Vertices are the 1st and 5th segments (indices 0 and 4)
 * 
 * @param orientation - "H" for horizontal, "V" for vertical
 * @param logicalRow - Wall row in logical grid (0-7)
 * @param logicalCol - Wall column in logical grid (0-7)
 * @returns Array of 2 vertex cells [start, end]
 */
export function wallVertices(
  orientation: 'H' | 'V',
  logicalRow: number,
  logicalCol: number
): [ExpandedCell, ExpandedCell] {
  const cells = wallToExpanded(orientation, logicalRow, logicalCol);
  return [cells[0], cells[4]];
}

/**
 * Check if two walls would cross (share midpoint - ILLEGAL)
 * 
 * @param wall1 - First wall
 * @param wall2 - Second wall
 * @returns True if walls cross (illegal placement)
 */
export function wallsCross(wall1: LogicalWall, wall2: LogicalWall): boolean {
  // Walls can only cross if they have different orientations
  if (wall1.orientation === wall2.orientation) {
    return false;
  }

  const mid1 = wallMidpoint(wall1.orientation, wall1.row, wall1.col);
  const mid2 = wallMidpoint(wall2.orientation, wall2.row, wall2.col);

  return mid1.row === mid2.row && mid1.col === mid2.col;
}

/**
 * Check if two walls overlap (share at least one edge segment - ILLEGAL)
 * 
 * @param wall1 - First wall
 * @param wall2 - Second wall
 * @returns True if walls overlap (illegal placement)
 */
export function wallsOverlap(wall1: LogicalWall, wall2: LogicalWall): boolean {
  // Walls can only overlap if they have the same orientation
  if (wall1.orientation !== wall2.orientation) {
    return false;
  }

  const cells1 = wallToExpanded(wall1.orientation, wall1.row, wall1.col);
  const cells2 = wallToExpanded(wall2.orientation, wall2.row, wall2.col);

  // Check if any cells are shared
  for (const cell1 of cells1) {
    for (const cell2 of cells2) {
      if (cell1.row === cell2.row && cell1.col === cell2.col) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Check if two walls are connected (touch at vertex or midpoint - LEGAL)
 * 
 * @param wall1 - First wall
 * @param wall2 - Second wall
 * @returns True if walls are connected (legal placement)
 */
export function wallsConnected(wall1: LogicalWall, wall2: LogicalWall): boolean {
  const cells1 = wallToExpanded(wall1.orientation, wall1.row, wall1.col);
  const cells2 = wallToExpanded(wall2.orientation, wall2.row, wall2.col);

  // Check if any endpoint or midpoint touches
  for (const cell1 of cells1) {
    for (const cell2 of cells2) {
      if (cell1.row === cell2.row && cell1.col === cell2.col) {
        return true;
      }
    }
  }

  return false;
}

/**
 * Convert display coordinates (1-based) to logical coordinates (0-based)
 * Used for human-readable coordinate entry
 * 
 * @param displayRow - Display row (1-9 for pawns, 1-8 for walls)
 * @param displayCol - Display column (1-9 for pawns, 1-8 for walls)
 * @returns Logical coordinates
 */
export function displayToLogical(displayRow: number, displayCol: number): ExpandedCell {
  return {
    row: displayRow - 1,
    col: displayCol - 1,
  };
}

/**
 * Convert logical coordinates (0-based) to display coordinates (1-based)
 * Used for human-readable coordinate display
 * 
 * @param logicalRow - Logical row (0-8 for pawns, 0-7 for walls)
 * @param logicalCol - Logical column (0-8 for pawns, 0-7 for walls)
 * @returns Display coordinates
 */
export function logicalToDisplay(logicalRow: number, logicalCol: number): ExpandedCell {
  return {
    row: logicalRow + 1,
    col: logicalCol + 1,
  };
}

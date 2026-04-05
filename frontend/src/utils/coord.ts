// coord.ts
// Utility functions for coordinate operations
// Responsibilities:
// - Coordinate transformations
// - Coordinate validation
// Forbidden:
// - No game logic
// - No backend communication

export type Coord = [number, number];

export function coordEquals(a: Coord, b: Coord): boolean {
  return a[0] === b[0] && a[1] === b[1];
}

export function coordToString(coord: Coord): string {
  return `${coord[0]},${coord[1]}`;
}

export function stringToCoord(s: string): Coord {
  const parts = s.split(",");
  return [parseInt(parts[0], 10), parseInt(parts[1], 10)];
}

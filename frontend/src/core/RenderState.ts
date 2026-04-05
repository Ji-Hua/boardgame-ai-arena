// RenderState.ts
//
// Pure projection of backend state for UI rendering.
// All rule-level identities use seat = 1 | 2.
// No agent/replay fields — human-vs-human only.

import type { Action } from "../types/Action";

export type Seat = 1 | 2;

/** Game result when terminal state is reached. */
export interface RenderResult {
  winnerSeat: Seat | null;
  termination: "goal" | "surrender" | "timeout" | "forced";
  surrenderBySeat: Seat | null;
}

export interface RenderPawnMap {
  1: [number, number];
  2: [number, number];
}

export interface RenderWall {
  orientation: "H" | "V";
  row: number;
  col: number;
  owner?: Seat;
}

/** Pure rendering interface — projected from backend state. */
export interface RenderState {
  boardSize: number;
  pawns: RenderPawnMap;
  walls: RenderWall[];
  wallsRemaining: { 1: number; 2: number };
  currentSeat: Seat;
  stepCount: number;
  lastAction: Action | null;
  /** null = no actor info needed for human-vs-human */
  actor: null;
  legalActions?: Action[];
  isTerminal: boolean;
  result: RenderResult | null;
}

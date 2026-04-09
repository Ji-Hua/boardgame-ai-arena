// RenderState.ts
//
// Pure projection of backend state for UI rendering.
// All rule-level identities use seat = 1 | 2.

import type { Action } from "../types/Action";

export type Seat = 1 | 2;

/** Actor type per seat. */
export type ActorType = "human" | "agent";

/** Per-seat actor configuration. */
export interface SeatActors {
  1: ActorType;
  2: ActorType;
}

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
  /** Per-seat actor types. null for legacy compatibility. */
  actors: SeatActors | null;
  /** Optional per-seat agent display names when a seat is controlled by an agent. */
  agentNames?: { 1: string | null; 2: string | null } | null;
  legalActions?: Action[];
  isTerminal: boolean;
  result: RenderResult | null;
}

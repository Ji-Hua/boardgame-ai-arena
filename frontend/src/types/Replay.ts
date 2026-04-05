// Replay.ts
// Type definitions for replay data model.
// See: documents/frontend/replay-frontend.md

import type { Action } from "./Action";

/** A single step in the replay sequence. */
export interface ReplayStep {
  /** Step identifier from source (e.g. "1", "6.1", "61.3") */
  stepId: string;

  /** The action attempted */
  action: Action;

  /** Whether the engine accepted or rejected this action */
  outcome: "accept" | "reject";
}

/** Complete replay dataset parsed from a replay source. */
export interface ReplayData {
  boardSize: number;
  initialPawns: { 1: [number, number]; 2: [number, number] };
  initialWallsRemaining: { 1: number; 2: number };
  steps: ReplayStep[];
}

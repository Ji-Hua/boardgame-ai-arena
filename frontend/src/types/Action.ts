// Action.ts
// Type definitions for backend Action schema
// This matches the backend contract: quoridor-protocol/contracts/core/action.schema.yaml
//
// Protocol v0.2.0: Unified Action Triplet
// - seat: 1 | 2 (rule-level identity)
// - type: "pawn" | "horizontal" | "vertical"
// - position: [row, col] in logical coordinates

export type Seat = 1 | 2;

/**
 * Unified Action Triplet (protocol v0.2.0)
 * All actions use this format regardless of source (REST, WebSocket, replay)
 */
export interface Action {
  seat: Seat;
  type: "pawn" | "horizontal" | "vertical";
  position: [number, number];
}

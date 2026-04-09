// StateMapper.ts
//
// Pure projection: backend wire format → RenderState.
// No Display Grid — the new backend uses logical coordinates directly.

import type { GameStateWire } from "../api/roomAPI";
import type { Action } from "../types/Action";
import type { Seat } from "../types/Action";
import type { RenderState, RenderWall, RenderResult, SeatActors } from "./RenderState";

export interface StateMapperParams {
  /** Backend game state in wire format (logical coordinates). */
  gameState: GameStateWire;

  /** Last action taken (already converted to Action). */
  lastAction: Action | null;

  /** Game result (only present when terminal). */
  result?: RenderResult | null;

  /** Accumulated wall placements tracked by the controller. */
  walls: RenderWall[];

  /** Step count from backend. */
  stepCount: number;

  /** Per-seat actor types (null for legacy human-vs-human). */
  actors?: SeatActors | null;
  /** Optional per-seat agent display names (when seats are agents). */
  agentNames?: { 1: string | null; 2: string | null } | null;
}

export class StateMapper {
  /**
   * Project backend wire state into a RenderState for the UI.
   *
   * The new backend returns pawns as { row, col }.
   * Wall positions are tracked by the controller (backend doesn't include them
   * in its serialised state yet).
   */
  static toRenderState(params: StateMapperParams): RenderState {
    const { gameState, lastAction, result, walls, stepCount, actors } = params;

    const { agentNames } = params;

    // Backend wire format: {row: engine_x, col: engine_y}
    // RenderState convention: [engine_y, engine_x] (row=y, col=x)
    const pawns = {
      1: [gameState.pawns["1"].col, gameState.pawns["1"].row] as [number, number],
      2: [gameState.pawns["2"].col, gameState.pawns["2"].row] as [number, number],
    };

    const wallsRemaining = {
      1: gameState.walls_remaining["1"],
      2: gameState.walls_remaining["2"],
    };

    const currentSeat: Seat = gameState.current_player;

    const isTerminal =
      gameState.game_over || (result !== null && result !== undefined);

    return {
      boardSize: 9,
      pawns,
      walls,
      wallsRemaining,
      currentSeat,
      stepCount,
      lastAction,
      actors: actors ?? null,
      agentNames: agentNames ?? null,
      isTerminal,
      result: result ?? null,
    };
  }
}

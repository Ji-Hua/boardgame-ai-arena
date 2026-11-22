// ActionTranslator.ts
// Responsibilities:
// - Converts UI events to backend Action schema
// - Translates frontend interaction format to backend contract
// - Must match backend Action schema: quoridor-protocol/contracts/core/action.schema.yaml
// Forbidden:
// - No backend communication
// - No state management
// - No game logic validation

import { Action } from "../types/Action";

export interface UIPawnMoveEvent {
  type: "pawn_move";
  player: 1 | 2;
  target: [number, number];
}

export interface UIWallPlacementEvent {
  type: "wall_placement";
  player: 1 | 2;
  orientation: "H" | "V";
  target: [number, number];
}

export type UIEvent = UIPawnMoveEvent | UIWallPlacementEvent;

export class ActionTranslator {
  /**
   * Translate pawn move UI event to backend Action
   */
  static translatePawnMove(
    player: 1 | 2,
    target: [number, number]
  ): Action {
    return {
      seat: player,
      type: "pawn",
      position: target,
    };
  }

  /**
   * Translate wall placement UI event to backend Action
   */
  static translateWallPlacement(
    player: 1 | 2,
    orientation: "H" | "V",
    target: [number, number]
  ): Action {
    return {
      seat: player,
      type: orientation === "H" ? "horizontal" : "vertical",
      position: target,
    };
  }

  /**
   * Translate generic UI event to backend Action
   */
  static translateUIEvent(event: UIEvent): Action {
    if (event.type === "pawn_move") {
      return this.translatePawnMove(event.player, event.target);
    } else {
      return this.translateWallPlacement(
        event.player,
        event.orientation,
        event.target
      );
    }
  }
}

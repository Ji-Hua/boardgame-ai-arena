// replayParser.ts
//
// Parses full_game_replay.md into ReplayData.
// The source uses (x, y) notation; this parser converts to [row, col] = [y, x].

import type { Action, Seat } from "../../types/Action";
import type { ReplayStep, ReplayData } from "../../types/Replay";

/**
 * Parse the markdown text of full_game_replay.md into structured ReplayData.
 */
export function parseReplayMarkdown(markdown: string): ReplayData {
  const steps: ReplayStep[] = [];

  // Match step blocks: "Step N:" or "Step N.M:"
  // followed by "- Action: { ... } -> ACCEPT|REJECT"
  const stepPattern =
    /Step\s+([\d.]+):\s*\n-\s*Action:\s*\{([^}]+)\}\s*->\s*(ACCEPT|REJECT)/g;

  let match: RegExpExecArray | null;
  while ((match = stepPattern.exec(markdown)) !== null) {
    const stepId = match[1];
    const actionBody = match[2];
    const outcome = match[3].toLowerCase() as "accept" | "reject";

    const action = parseActionBody(actionBody);
    steps.push({ stepId, action, outcome });
  }

  return {
    boardSize: 9,
    initialPawns: {
      1: [0, 4], // source (4,0) → [row=0, col=4]
      2: [8, 4], // source (4,8) → [row=8, col=4]
    },
    initialWallsRemaining: { 1: 10, 2: 10 },
    steps,
  };
}

/**
 * Parse the inner body of an action: "player: 1, kind: MovePawn, target: (4,1,Square)"
 */
function parseActionBody(body: string): Action {
  // Extract player
  const playerMatch = body.match(/player:\s*(\d)/);
  if (!playerMatch) throw new Error(`Cannot parse player from: ${body}`);
  const seat = parseInt(playerMatch[1], 10) as Seat;

  // Extract kind
  const kindMatch = body.match(/kind:\s*(\w+)/);
  if (!kindMatch) throw new Error(`Cannot parse kind from: ${body}`);
  const kind = kindMatch[1];

  // Extract target: (x, y, Type)
  const targetMatch = body.match(/target:\s*\((-?\d+)\s*,\s*(-?\d+)\s*,\s*(\w+)\)/);
  if (!targetMatch) throw new Error(`Cannot parse target from: ${body}`);
  const x = parseInt(targetMatch[1], 10);
  const y = parseInt(targetMatch[2], 10);
  const targetType = targetMatch[3]; // Square, Horizontal, Vertical

  // Convert (x, y) → [row, col] = [y, x]
  const position: [number, number] = [y, x];

  // Map kind + targetType to action type
  let type: Action["type"];
  if (kind === "MovePawn") {
    type = "pawn";
  } else if (kind === "PlaceWall") {
    if (targetType === "Horizontal") {
      type = "horizontal";
    } else if (targetType === "Vertical") {
      type = "vertical";
    } else {
      throw new Error(`Unknown wall type: ${targetType}`);
    }
  } else {
    throw new Error(`Unknown action kind: ${kind}`);
  }

  return { seat, type, position };
}

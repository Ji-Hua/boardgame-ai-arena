# Quoridor Frontend Replay Specification

Author: Ji Hua  
Created Date: 2026-04-05  
Last Modified: 2026-04-05  
Current Version: 1  
Document Type: Design  
Document Subtype: Replay Specification  
Document Status: In Development  
Document Authority Scope: Frontend module  
Document Purpose:  
This document defines how the frontend replays a recorded game from a structured action sequence. It covers the replay data model, parsing strategy, execution model, and integration with the existing render-driven architecture.

---

# 1. Replay Data Model

## 1.1 ReplayStep

A single step in the replay sequence.

```typescript
interface ReplayStep {
  /** Step identifier in replay source (e.g. "1", "6.1", "61.3") */
  stepId: string;

  /** The action attempted */
  action: Action;

  /** Whether the action was accepted or rejected */
  outcome: "accept" | "reject";
}
```

- `action` uses the standard `Action` type: `{ seat: 1 | 2, type: "pawn" | "horizontal" | "vertical", position: [row, col] }`
- `position` is in frontend logical coordinates: `[row, col]` where row is the y-axis and col is the x-axis.

## 1.2 ReplayData

The full replay dataset.

```typescript
interface ReplayData {
  /** Board size (always 9) */
  boardSize: number;

  /** Initial pawn positions: seat → [row, col] */
  initialPawns: { 1: [number, number]; 2: [number, number] };

  /** Initial walls remaining per seat */
  initialWallsRemaining: { 1: number; 2: number };

  /** Ordered sequence of all steps (both accept and reject) */
  steps: ReplayStep[];
}
```

## 1.3 Coordinate Mapping

The replay source `full_game_replay.md` uses `(x, y)` notation where:
- `x` = column (file), `y` = row (rank)
- Player 1 starts at `(4, 0)` → frontend `[row=0, col=4]`
- Player 2 starts at `(4, 8)` → frontend `[row=8, col=4]`

Conversion: source `(x, y)` → frontend `[y, x]` i.e. `[row, col]`.

This applies to both pawn targets and wall positions.

---

# 2. Parsing Strategy

## 2.1 Source Format

`full_game_replay.md` is a Markdown document with steps in this format:

```
Step N:
- Action: { player: P, kind: MovePawn|PlaceWall, target: (x, y, Type) } -> ACCEPT|REJECT
```

Where:
- `N` is an integer (accepted step) or decimal (rejected step, e.g. `6.1`)
- `player` is `1` or `2`
- `kind` is `MovePawn` or `PlaceWall`
- `target` is `(x, y, Square)` for pawn moves or `(x, y, Horizontal|Vertical)` for walls
- Result is `ACCEPT` or `REJECT`

## 2.2 Parsing Rules

1. Read the file as a string.
2. Match each `Step N:` block using regex.
3. For each step, extract:
   - `stepId` from the step number
   - `player` → maps to `seat`
   - `kind` + `target` type → maps to `action.type`:
     - `MovePawn` → `"pawn"`
     - `PlaceWall` + `Horizontal` → `"horizontal"`
     - `PlaceWall` + `Vertical` → `"vertical"`
   - `(x, y)` → maps to `position: [y, x]` (swap x/y)
   - `ACCEPT` / `REJECT` → maps to `outcome`

4. Collect all steps in document order into `ReplayData.steps`.

## 2.3 Validation

After parsing, validate:
- All ACCEPT steps have integer step IDs
- All REJECT steps have decimal step IDs
- Step sequence is monotonically ordered
- Initial state matches the header (Player 1 at (4,0), Player 2 at (4,8), 10 walls each)

---

# 3. Replay Execution Model

## 3.1 Game State

The replay maintains a mutable game state that evolves through ACCEPT steps:

```typescript
interface ReplayGameState {
  pawns: { 1: [number, number]; 2: [number, number] };
  walls: RenderWall[];
  wallsRemaining: { 1: number; 2: number };
  currentPlayer: 1 | 2;
  gameOver: boolean;
  winner: 1 | 2 | null;
}
```

## 3.2 Step Execution

For each step in the replay:

### ACCEPT Steps

1. Apply the action to the current `ReplayGameState`:
   - **Pawn move**: Update `pawns[seat]` to `action.position`
   - **Wall placement**: Add wall to `walls` array, decrement `wallsRemaining[seat]`
2. Switch `currentPlayer` to the other seat.
3. Check win condition: if pawn reaches target row (seat 1 → row 8, seat 2 → row 0), set `gameOver = true` and `winner`.
4. Project state through `StateMapper` → `RenderState` → `renderStore`.

### REJECT Steps

1. Do NOT modify `ReplayGameState`.
2. Project the current (unchanged) state to `RenderState`.
3. The UI should show the rejected action visually (optional: highlight the attempted move as rejected).

## 3.3 Stepping Modes

The ReplayController supports:

- `stepForward()` — advance to the next step
- `stepBackward()` — go back to the previous step
- `jumpToStep(index)` — jump to any step index
- `jumpToStart()` — reset to initial state (before step 0)
- `jumpToEnd()` — advance to final state

For backward navigation, the controller recomputes state from the initial state up to the target index by replaying all ACCEPT steps up to that point. This avoids maintaining undo history.

## 3.4 State Recomputation

```
Given target step index N:
1. Start from initial state
2. For each step[i] where i <= N:
   - If step[i].outcome === "accept":
     - Apply action to state
3. Project final state to RenderState
```

This is O(N) for each navigation but ensures correctness without inverse operations.

---

# 4. Integration with Frontend Architecture

## 4.1 Data Flow

```
ReplayData → ReplayController → StateMapper → RenderState → renderStore → Board
```

This matches the existing render-driven architecture. The ReplayController replaces LiveController as the data source, but the downstream pipeline is identical.

## 4.2 StateMapper Integration

ReplayController produces `StateMapperParams` compatible with the existing `StateMapper.toRenderState()`:

```typescript
{
  gameState: {
    current_player: replayGameState.currentPlayer,
    pawns: {
      "1": { row: pawns[1][0], col: pawns[1][1] },
      "2": { row: pawns[2][0], col: pawns[2][1] },
    },
    walls_remaining: {
      "1": replayGameState.wallsRemaining[1],
      "2": replayGameState.wallsRemaining[2],
    },
    game_over: replayGameState.gameOver,
    winner: replayGameState.winner,
  },
  lastAction: step.outcome === "accept" ? step.action : null,
  result: gameOver ? { winnerSeat: winner, termination: "goal", surrenderBySeat: null } : null,
  walls: replayGameState.walls,
  stepCount: acceptedStepCount,
}
```

The `gameState` field conforms to `GameStateWire` from `roomAPI.ts`.

## 4.3 renderStore

ReplayController writes to the same `renderStore` used by LiveController. The Board component subscribes to `renderStore` and renders identically regardless of whether the state came from live play or replay.

## 4.4 Board Rendering

- Board renders in `mode="replay"` (non-interactive) or `mode="play"` with interactions disabled.
- Pawns, walls, and wall stocks display from RenderState.
- No interaction layer is active during replay.

---

# 5. Constraints

1. **No rule logic in frontend**: The ReplayController applies actions mechanically (update coordinates, add walls). It does NOT validate legality. The replay data's `outcome` field determines whether an action is applied.

2. **No state bypass**: All rendering goes through `StateMapper → RenderState → renderStore → Board`. The replay MUST NOT directly manipulate DOM or component state.

3. **Architecture compliance**: The replay module follows `frontend-architecture.md` — it is a controller that writes to `renderStore`, just like `LiveController`.

4. **Isolation from live mode**: Replay code is contained in `src/modes/replay/`. It does not import from or depend on `LiveController` or `websocket.ts`. The only shared dependencies are `StateMapper`, `RenderState`, `renderStore`, and `Action`.

5. **Replay does not infer state**: The replay controller does not compute legal moves, validate paths, or determine winners independently. It trusts the replay data's outcome annotations.

---

# 6. Runtime Behavior in Browser

## 6.1 Data Source

The replay data is served as a pre-built JSON file from the frontend's public directory:

```
GET /replays/full_game_replay.json
```

The JSON file is generated offline from `documents/engine/implementation/full_game_replay.md` using
`scripts/replay_md_to_json.js`. It is committed to `frontend/public/replays/` and served as a static
asset by both the Vite dev server and the Vite preview server (Docker).

The JSON format matches `ReplayData` directly — no client-side markdown parsing is performed.

## 6.2 Data Flow (JSON → Render)

```
/replays/full_game_replay.json
  │  (fetch, JSON.parse)
  ▼
ReplayData  (loaded in App.tsx)
  │  (passed as prop)
  ▼
ReplayController  (src/modes/replay/ReplayController.ts)
  │  (stepForward / auto-play timer)
  ▼
ReplayGameState  (internal mutable state)
  │  (StateMapper.toRenderState)
  ▼
RenderState
  │  (renderStore.setState)
  ▼
renderStore  (src/stores/renderStore.ts)
  │  (React subscription)
  ▼
ReplayPage  (src/view/replay/ReplayPage.tsx)
  │  (props: pawns, walls)
  ▼
Board  (src/components/board/Board.tsx, mode="replay")
```

## 6.3 Timing Model

The replay auto-plays at 0.5 seconds per step using `setInterval`:

```
interval = 500ms
ACCEPT step → apply state change → advance cursor → render
REJECT step → do NOT change game state → advance cursor → render (state unchanged)
```

Both ACCEPT and REJECT steps advance the step cursor and consume one interval tick. The visual display
shows the `stepId`, action details, and outcome (`✓ ACCEPT` / `✗ REJECT`) for every step regardless
of outcome.

## 6.4 Auto-Play Controls

| Action | Behaviour |
|--------|-----------|
| Open Replay page | Auto-play starts immediately |
| Space | Toggle play / pause |
| ← → arrows | Step backward / forward (pauses auto-play) |
| Home / End | Jump to start / end (pauses auto-play) |
| Pause button | Stops timer |
| Play button (at end) | Restarts from beginning |

## 6.5 How to Trigger Replay in the UI

1. Open `http://localhost:8765` in a browser.
2. Click **📼 Replay Full Game** on the main menu.
3. The browser fetches `/replays/full_game_replay.json`.
4. The replay starts auto-playing immediately at 0.5s per step.
5. Use Space to pause, arrow keys to step manually.

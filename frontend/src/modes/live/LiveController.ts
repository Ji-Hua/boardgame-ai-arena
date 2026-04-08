// LiveController.ts
//
// The ONLY module responsible for backend communication.
// Implements REST bootstrap + WebSocket runtime per frontend-architecture.md.
//
// Local mode: one client controls both seats via a single WebSocket connection.
// WebSocket state_update is the ONLY authoritative runtime progress signal.

import { roomAPI, type GameStateWire } from "../../api/roomAPI";
import { WebSocketClient, type StateUpdateEvent, type ActionWire } from "../../api/websocket";
import { renderStore } from "../../stores/renderStore";
import { StateMapper } from "../../core/StateMapper";
import type { Action, Seat } from "../../types/Action";
import type { RenderWall, RenderResult, SeatActors, ActorType } from "../../core/RenderState";

/** Configuration for a game session. */
export interface GameConfig {
  seat1: ActorType;
  seat2: ActorType;
  /** Agent type for seat 1 (required when seat1 is "agent"). */
  agent1Type?: string;
  /** Agent type for seat 2 (required when seat2 is "agent"). */
  agent2Type?: string;
  /** Optional agent config for seat 1. */
  agent1Config?: Record<string, unknown>;
  /** Optional agent config for seat 2. */
  agent2Config?: Record<string, unknown>;
}

export class LiveController {
  private roomId: string | null = null;
  private gameId: string | null = null;
  private clientId: string;
  private wsClient: WebSocketClient | null = null;
  private config: GameConfig = { seat1: "human", seat2: "human" };

  // Per-seat actor types for rendering
  public actors: SeatActors = { 1: "human", 2: "human" };

  // Accumulated walls (backend GameState may not include wall positions yet)
  private walls: RenderWall[] = [];

  // Game end state
  public gameEnded: boolean = false;
  public gameResult: RenderResult | null = null;

  // Error message for UI
  public lastError: string | null = null;

  // Simple observer pattern for UI components
  private listeners: Set<() => void> = new Set();

  constructor() {
    this.clientId = this.getOrCreateClientId();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /** Update game speed on the backend. Only affects agent-vs-agent / replay modes. */
  async setGameSpeed(multiplier: number): Promise<void> {
    if (!this.roomId) return;
    try {
      await roomAPI.setGameSpeed(this.roomId, multiplier);
    } catch (err) {
      console.warn("setGameSpeed failed:", err);
    }
  }

  private notify(): void {
    this.listeners.forEach((l) => l());
  }

  // ── Bootstrap (REST) ─────────────────────────────────────────

  /**
   * Full bootstrap sequence:
   * 1. Create room
   * 2. Join both seats with same client_id
   * 3. Select actor for both seats
   * 4. Create agents for agent-controlled seats
   * 5. Start game
   * 6. Connect WebSocket and subscribe
   *
   * Returns the room_id.
   */
  async bootstrap(config?: GameConfig): Promise<string> {
    if (config) {
      this.config = config;
    }
    this.actors = { 1: this.config.seat1, 2: this.config.seat2 };

    // 1. Create room
    const room = await roomAPI.createRoom();
    this.roomId = room.room_id;

    // 2. Join both seats (local mode: same browser, two logical clients)
    await roomAPI.joinSeat(this.roomId, this.clientId, 1);
    await roomAPI.joinSeat(this.roomId, this.clientId + "-2", 2);

    // 3. Select actor type for both seats
    await roomAPI.selectActor(this.roomId, 1, this.config.seat1);
    await roomAPI.selectActor(this.roomId, 2, this.config.seat2);

    // 4. Create agents for agent-controlled seats
    if (this.config.seat1 === "agent" && this.config.agent1Type) {
      await roomAPI.createAgent(
        this.roomId, 1, this.config.agent1Type, this.config.agent1Config,
      );
    }
    if (this.config.seat2 === "agent" && this.config.agent2Type) {
      await roomAPI.createAgent(
        this.roomId, 2, this.config.agent2Type, this.config.agent2Config,
      );
    }

    // 5. Start game
    const startResp = await roomAPI.startGame(this.roomId);
    this.gameId = startResp.game.game_id;

    // Render initial state from start_game response
    this.projectState(startResp.game.state, null, 0);

    // 6. Connect WebSocket
    await this.connectWebSocket(this.roomId);

    return this.roomId;
  }

  // ── Actions (via WebSocket) ──────────────────────────────────

  /**
   * Send a take_action message via WebSocket.
   * The action uses the current seat from RenderState.
   */
  takeAction(action: ActionWire): void {
    if (!this.wsClient) {
      console.error("WebSocket not connected");
      return;
    }
    // Wall tracking is done only on confirmed state_update (handleStateUpdate).
    // Do NOT push optimistically here — rejected walls would never be removed.
    this.wsClient.takeAction(action);
  }

  /**
   * Send a take_action and return a promise that resolves
   * with the next action_result. Used by replay for sequential playback.
   */
  takeActionAsync(action: ActionWire): Promise<{ success: boolean; error?: string }> {
    return new Promise((resolve) => {
      if (!this.wsClient) {
        resolve({ success: false, error: "WebSocket not connected" });
        return;
      }

      // One-shot listener for action_result
      const unsub = this.wsClient!.onActionResult((event) => {
        unsub();
        resolve({ success: event.success, error: event.error ?? undefined });
      });

      this.wsClient.takeAction(action);
    });
  }

  /**
   * Surrender the game for the given seat.
   */
  surrender(seat: 1 | 2): void {
    if (!this.wsClient) return;
    this.wsClient.surrender(seat);
  }

  /**
   * Fetch legal pawn moves for the current player from backend.
   * Returns a promise of [engine_y, engine_x] pairs (RenderState convention).
   */
  getLegalPawnMoves(): Promise<[number, number][]> {
    return new Promise((resolve) => {
      if (!this.wsClient) {
        resolve([]);
        return;
      }
      const unsub = this.wsClient.onLegalActionsResult((event) => {
        unsub();
        // Backend target = [engine_x, engine_y]; RenderState uses [engine_y, engine_x]
        const moves: [number, number][] = event.actions.map(
          (a) => [a.target[1], a.target[0]]
        );
        resolve(moves);
      });
      this.wsClient.getLegalActions();
    });
  }

  /**
   * Disconnect and clean up all resources.
   */
  disconnect(): void {
    if (this.wsClient) {
      this.wsClient.close();
      this.wsClient = null;
    }
    this.roomId = null;
    this.gameId = null;
    this.walls = [];
    this.gameEnded = false;
    this.gameResult = null;
    this.lastError = null;
    this.config = { seat1: "human", seat2: "human" };
    this.actors = { 1: "human", 2: "human" };
  }

  // ── WebSocket ────────────────────────────────────────────────

  private async connectWebSocket(roomId: string): Promise<void> {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    let wsHost: string;
    const port = window.location.port;
    if (port === "8765") {
      wsHost = `${window.location.hostname}:8764`;
    } else {
      wsHost = window.location.host;
    }
    const wsUrl = `${protocol}//${wsHost}/ws/${roomId}`;

    this.wsClient = new WebSocketClient(wsUrl);
    await this.wsClient.connect();
    this.wsClient.subscribe(this.clientId);

    // state_update — the ONLY authoritative runtime signal
    this.wsClient.onStateUpdate((event: StateUpdateEvent) => {
      this.handleStateUpdate(event);
    });

    // action_result — ACK only, not state authority
    this.wsClient.onActionResult((event) => {
      if (!event.success) {
        this.lastError = event.error ?? "Action rejected";
        this.notify();
      }
    });

    // game_ended
    this.wsClient.onGameEnded((event) => {
      const result: RenderResult = {
        winnerSeat: event.result.winner_seat,
        termination: event.result.termination,
        surrenderBySeat: null,
      };
      this.handleTerminal(result);
    });

    // errors
    this.wsClient.onError((event) => {
      this.lastError = event.message;
      this.notify();
    });
  }

  private handleStateUpdate(event: StateUpdateEvent): void {
    this._applyStateUpdate(event);
  }

  private _applyStateUpdate(event: StateUpdateEvent): void {
    // Track wall from last_action
    if (event.last_action) {
      const la = event.last_action;
      // la.target = [x, y] from backend; store as {row: y, col: x}
      if (la.type === "horizontal" || la.type === "vertical") {
        // Only add if not already tracked (avoid duplicates from our own takeAction)
        const exists = this.walls.some(
          (w) =>
            w.orientation === (la.type === "horizontal" ? "H" : "V") &&
            w.row === la.target[1] &&
            w.col === la.target[0],
        );
        if (!exists) {
          this.walls.push({
            orientation: la.type === "horizontal" ? "H" : "V",
            row: la.target[1],
            col: la.target[0],
            owner: la.player,
          });
        }
      }
    }

    const lastAction: Action | null = event.last_action
      ? {
          seat: event.last_action.player,
          type: event.last_action.type,
          position: event.last_action.target,
        }
      : null;

    this.projectState(event.state, lastAction, event.step_count);
    this.notify();
  }

  // ── State projection ─────────────────────────────────────────

  private projectState(
    gameState: GameStateWire,
    lastAction: Action | null,
    stepCount: number,
    result?: RenderResult | null,
  ): void {
    const renderState = StateMapper.toRenderState({
      gameState,
      lastAction,
      result: result ?? null,
      walls: this.walls,
      stepCount,
      actors: this.actors,
    });
    renderStore.setState(renderState);
  }

  private handleTerminal(result: RenderResult): void {
    this.gameEnded = true;
    this.gameResult = result;

    const currentState = renderStore.getState();
    renderStore.setState({
      ...currentState,
      isTerminal: true,
      result,
    });
    this.notify();
  }

  // ── Utilities ────────────────────────────────────────────────

  private getOrCreateClientId(): string {
    let id = sessionStorage.getItem("quoridor_client_id");
    if (!id) {
      id = crypto.randomUUID
        ? crypto.randomUUID()
        : "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
            const r = (Math.random() * 16) | 0;
            const v = c === "x" ? r : (r & 0x3) | 0x8;
            return v.toString(16);
          });
      sessionStorage.setItem("quoridor_client_id", id);
    }
    return id;
  }
}

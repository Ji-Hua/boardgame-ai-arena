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
import type { RenderWall, RenderResult } from "../../core/RenderState";

export class LiveController {
  private roomId: string | null = null;
  private gameId: string | null = null;
  private clientId: string;
  private wsClient: WebSocketClient | null = null;

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

  private notify(): void {
    this.listeners.forEach((l) => l());
  }

  // ── Bootstrap (REST) ─────────────────────────────────────────

  /**
   * Full local-mode bootstrap sequence:
   * 1. Create room
   * 2. Join both seats with same client_id
   * 3. Select actor for both seats (human)
   * 4. Start game
   * 5. Connect WebSocket and subscribe
   *
   * Returns the room_id.
   */
  async bootstrap(): Promise<string> {
    // 1. Create room
    const room = await roomAPI.createRoom();
    this.roomId = room.room_id;

    // 2. Join both seats (local mode: same browser, two logical clients)
    await roomAPI.joinSeat(this.roomId, this.clientId, 1);
    await roomAPI.joinSeat(this.roomId, this.clientId + "-2", 2);

    // 3. Select actor type for both seats
    await roomAPI.selectActor(this.roomId, 1, "human");
    await roomAPI.selectActor(this.roomId, 2, "human");

    // 4. Start game
    const startResp = await roomAPI.startGame(this.roomId);
    this.gameId = startResp.game.game_id;

    // Render initial state from start_game response
    this.projectState(startResp.game.state, null, 0);

    // 5. Connect WebSocket
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

    // Track wall ownership for rendering
    // action.target = [x, y] (backend convention); store as {row: y, col: x}
    if (action.type === "horizontal" || action.type === "vertical") {
      const seat = action.player;
      this.walls.push({
        orientation: action.type === "horizontal" ? "H" : "V",
        row: action.target[1],
        col: action.target[0],
        owner: seat,
      });
    }

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

      // Track wall ownership (same as takeAction)
      if (action.type === "horizontal" || action.type === "vertical") {
        const seat = action.player;
        this.walls.push({
          orientation: action.type === "horizontal" ? "H" : "V",
          row: action.target[1],
          col: action.target[0],
          owner: seat,
        });
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

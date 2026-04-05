// websocket.ts
// WebSocket client for the new backend room-based protocol.
// Handles subscribe, take_action, and incoming events.
// This is the only runtime communication channel.

import type { GameStateWire } from "./roomAPI";

// ── Event types ────────────────────────────────────────────────

export interface ActionWire {
  player: 1 | 2;
  type: "pawn" | "horizontal" | "vertical";
  target: [number, number];
}

export interface RoomSnapshotEvent {
  type: "room_snapshot";
  room_id: string;
  status: "config" | "using";
  seats: {
    "1": { client_id: string | null; actor_type: string | null };
    "2": { client_id: string | null; actor_type: string | null };
  };
  game: {
    game_id: string | null;
    phase: string | null;
    state: GameStateWire | null;
  };
}

export interface StateUpdateEvent {
  type: "state_update";
  game_id: string;
  state: GameStateWire;
  last_action: ActionWire;
  step_count: number;
}

export interface ActionResultEvent {
  type: "action_result";
  success: boolean;
  error: string | null;
}

export interface GameStartedEvent {
  type: "game_started";
  game_id: string;
  state: GameStateWire;
}

export interface GameEndedEvent {
  type: "game_ended";
  game_id: string;
  result: {
    winner_seat: 1 | 2 | null;
    termination: "goal" | "surrender" | "forced";
  };
}

export interface WsErrorEvent {
  type: "error";
  code: string;
  message: string;
}

// ── Handler types ──────────────────────────────────────────────

type RoomSnapshotHandler = (event: RoomSnapshotEvent) => void;
type StateUpdateHandler = (event: StateUpdateEvent) => void;
type ActionResultHandler = (event: ActionResultEvent) => void;
type GameStartedHandler = (event: GameStartedEvent) => void;
type GameEndedHandler = (event: GameEndedEvent) => void;
type ErrorHandler = (event: WsErrorEvent) => void;

// ── WebSocket client ───────────────────────────────────────────

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private wsUrl: string;

  private roomSnapshotHandlers: Set<RoomSnapshotHandler> = new Set();
  private stateUpdateHandlers: Set<StateUpdateHandler> = new Set();
  private actionResultHandlers: Set<ActionResultHandler> = new Set();
  private gameStartedHandlers: Set<GameStartedHandler> = new Set();
  private gameEndedHandlers: Set<GameEndedHandler> = new Set();
  private errorHandlers: Set<ErrorHandler> = new Set();

  constructor(wsUrl: string) {
    this.wsUrl = wsUrl;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        console.log("WebSocket connected");
        resolve();
      };

      this.ws.onerror = (event) => {
        console.error("WebSocket error:", event);
        this.errorHandlers.forEach((h) =>
          h({ type: "error", code: "WEBSOCKET_ERROR", message: "Connection error" }),
        );
        reject(new Error("WebSocket connection failed"));
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (err) {
          console.error("Failed to parse WebSocket message:", err);
        }
      };

      this.ws.onclose = () => {
        console.log("WebSocket disconnected");
      };
    });
  }

  private handleMessage(message: unknown): void {
    if (!message || typeof message !== "object" || !("type" in message)) {
      console.warn("Invalid WebSocket message:", message);
      return;
    }

    const msg = message as { type: string };

    switch (msg.type) {
      case "room_snapshot":
        this.roomSnapshotHandlers.forEach((h) =>
          h(msg as unknown as RoomSnapshotEvent),
        );
        break;
      case "state_update":
        this.stateUpdateHandlers.forEach((h) =>
          h(msg as unknown as StateUpdateEvent),
        );
        break;
      case "action_result":
        this.actionResultHandlers.forEach((h) =>
          h(msg as unknown as ActionResultEvent),
        );
        break;
      case "game_started":
        this.gameStartedHandlers.forEach((h) =>
          h(msg as unknown as GameStartedEvent),
        );
        break;
      case "game_ended":
        this.gameEndedHandlers.forEach((h) =>
          h(msg as unknown as GameEndedEvent),
        );
        break;
      case "error":
        this.errorHandlers.forEach((h) =>
          h(msg as unknown as WsErrorEvent),
        );
        break;
      default:
        console.warn("Unknown WebSocket message type:", msg.type);
    }
  }

  // ── Client → Server ────────────────────────────────────────

  subscribe(clientId: string): void {
    this.send({ type: "subscribe", client_id: clientId });
  }

  takeAction(action: ActionWire): void {
    this.send({ type: "take_action", action });
  }

  surrender(seat: 1 | 2): void {
    this.send({ type: "surrender", seat });
  }

  // ── Event registration ─────────────────────────────────────

  onRoomSnapshot(handler: RoomSnapshotHandler): () => void {
    this.roomSnapshotHandlers.add(handler);
    return () => this.roomSnapshotHandlers.delete(handler);
  }

  onStateUpdate(handler: StateUpdateHandler): () => void {
    this.stateUpdateHandlers.add(handler);
    return () => this.stateUpdateHandlers.delete(handler);
  }

  onActionResult(handler: ActionResultHandler): () => void {
    this.actionResultHandlers.add(handler);
    return () => this.actionResultHandlers.delete(handler);
  }

  onGameStarted(handler: GameStartedHandler): () => void {
    this.gameStartedHandlers.add(handler);
    return () => this.gameStartedHandlers.delete(handler);
  }

  onGameEnded(handler: GameEndedHandler): () => void {
    this.gameEndedHandlers.add(handler);
    return () => this.gameEndedHandlers.delete(handler);
  }

  onError(handler: ErrorHandler): () => void {
    this.errorHandlers.add(handler);
    return () => this.errorHandlers.delete(handler);
  }

  // ── Lifecycle ──────────────────────────────────────────────

  close(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private send(message: unknown): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error("WebSocket is not connected");
      return;
    }
    this.ws.send(JSON.stringify(message));
  }
}

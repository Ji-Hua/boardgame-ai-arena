// roomAPI.ts
// REST API client for the new backend room endpoints.
// Used only during bootstrap (room creation, seat join, actor select, game start).
// All runtime communication goes through WebSocket.

function computeBaseUrl(): string {
  const envUrl =
    typeof window !== "undefined" && (window as any).VITE_API_BASE_URL
      ? (window as any).VITE_API_BASE_URL
      : undefined;
  if (envUrl && envUrl.trim().length > 0) return envUrl;

  try {
    const { protocol, hostname, port } = window.location;
    // Dev mode: frontend on 8765 → backend on 8764
    if (port === "8765") {
      return `${protocol}//${hostname}:8764/api`;
    }
    // Production: same origin
    return `${protocol}//${hostname}:${port}/api`;
  } catch {
    return "http://localhost:8764/api";
  }
}

const BASE_URL = computeBaseUrl();

// ── Response types ─────────────────────────────────────────────

export interface SeatInfo {
  client_id: string | null;
  actor_type: string | null;
}

export interface RoomSnapshot {
  room_id: string;
  status: "config" | "using" | "closed";
  seats: {
    "1": SeatInfo;
    "2": SeatInfo;
  };
}

export interface GameStateWire {
  current_player: 1 | 2;
  pawns: {
    "1": { row: number; col: number };
    "2": { row: number; col: number };
  };
  walls_remaining: {
    "1": number;
    "2": number;
  };
  game_over: boolean;
  winner: 1 | 2 | null;
}

export interface StartGameResponse {
  room_id: string;
  status: "using";
  game: {
    game_id: string;
    phase: "running";
    state: GameStateWire;
  };
}

// ── API client ─────────────────────────────────────────────────

async function post(url: string, body?: unknown): Promise<Response> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status}: ${text}`);
  }
  return res;
}

export const roomAPI = {
  /** POST /api/rooms → RoomSnapshot */
  async createRoom(): Promise<RoomSnapshot> {
    const res = await post(`${BASE_URL}/rooms`);
    return res.json();
  },

  /** POST /api/rooms/{room_id}/join */
  async joinSeat(
    roomId: string,
    clientId: string,
    seat: 1 | 2,
  ): Promise<RoomSnapshot> {
    const res = await post(`${BASE_URL}/rooms/${roomId}/join`, {
      client_id: clientId,
      seat,
    });
    return res.json();
  },

  /** POST /api/rooms/{room_id}/select_actor */
  async selectActor(
    roomId: string,
    seat: 1 | 2,
    actorType: "human" | "agent",
  ): Promise<RoomSnapshot> {
    const res = await post(`${BASE_URL}/rooms/${roomId}/select_actor`, {
      seat,
      actor_type: actorType,
    });
    return res.json();
  },

  /** POST /api/rooms/{room_id}/start_game → StartGameResponse */
  async startGame(roomId: string): Promise<StartGameResponse> {
    const res = await post(`${BASE_URL}/rooms/${roomId}/start_game`);
    return res.json();
  },

  /** POST /api/rooms/{room_id}/agent/create */
  async createAgent(
    roomId: string,
    seat: 1 | 2,
    agentType: string,
    config?: Record<string, unknown>,
  ): Promise<{ instance_id: string; room_id: string; seat: number }> {
    const body: Record<string, unknown> = { seat, agent_type: agentType };
    if (config !== undefined) body.config = config;
    const res = await post(`${BASE_URL}/rooms/${roomId}/agent/create`, body);
    return res.json();
  },

  /** POST /api/rooms/{room_id}/agent/start */
  async startAgents(roomId: string): Promise<{ room_id: string; started: unknown[] }> {
    const res = await post(`${BASE_URL}/rooms/${roomId}/agent/start`);
    return res.json();
  },

  /** POST /api/rooms/{room_id}/game/speed */
  async setGameSpeed(
    roomId: string,
    speedMultiplier: number,
  ): Promise<{ room_id: string; speed_multiplier: number }> {
    const res = await post(`${BASE_URL}/rooms/${roomId}/game/speed`, {
      speed_multiplier: speedMultiplier,
    });
    return res.json();
  },

  /** GET /api/agent/types */
  async getAgentTypes(): Promise<{ agent_types: AgentTypeInfo[] }> {
    const res = await fetch(`${BASE_URL}/agent/types`);
    if (!res.ok) throw new Error(`${res.status}`);
    return res.json();
  },
};

export interface AgentTypeInfo {
  type_id: string;
  display_name: string;
  category: string;
}

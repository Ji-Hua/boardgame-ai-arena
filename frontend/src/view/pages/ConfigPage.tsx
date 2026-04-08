// ConfigPage.tsx
// Pre-game configuration page — seat selection, agent types, replay setup.
// User configures both seats before starting the game.
// Game does NOT start until "Start Game" is clicked.

import React, { useState, useEffect } from "react";
import type { GameConfig } from "../../modes/live/LiveController";
import type { ActorType } from "../../core/RenderState";
import { roomAPI, type AgentTypeInfo } from "../../api/roomAPI";
import { PLAYER_COLORS } from "../../theme/playerColors";
import type { ReplayData } from "../../types/Replay";

interface ConfigPageProps {
  onStart: (config: GameConfig) => void;
  onBack: () => void;
}

interface SeatConfig {
  actorType: ActorType;
  agentType: string | null;
}

/** Convert frontend replay action to backend engine format. */
function convertReplayAction(action: { seat: number; type: string; position: [number, number] }): {
  player: number;
  type: string;
  target: [number, number];
} {
  // Frontend position: [y, x] (row, col) → Backend target: [x, y] (col, row)
  return {
    player: action.seat,
    type: action.type,
    target: [action.position[1], action.position[0]],
  };
}

export const ConfigPage: React.FC<ConfigPageProps> = ({ onStart, onBack }) => {
  const [agentTypes, setAgentTypes] = useState<AgentTypeInfo[]>([]);
  const [seat1, setSeat1] = useState<SeatConfig>({ actorType: "human", agentType: null });
  const [seat2, setSeat2] = useState<SeatConfig>({ actorType: "human", agentType: null });
  const [replayData, setReplayData] = useState<ReplayData | null>(null);
  const [replayLoading, setReplayLoading] = useState(false);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Fetch available agent types on mount
  useEffect(() => {
    roomAPI.getAgentTypes().then((resp) => {
      setAgentTypes(resp.agent_types);
    }).catch(() => {
      setError("Failed to fetch agent types");
    });
  }, []);

  // Non-replay agent types for the dropdown
  const nonReplayAgents = agentTypes.filter((a) => a.category !== "replay");
  const hasReplayAgent = agentTypes.some((a) => a.category === "replay");

  // Determine if either seat is set to replay
  const isSeat1Replay = seat1.actorType === "agent" && seat1.agentType === "replay";
  const isSeat2Replay = seat2.actorType === "agent" && seat2.agentType === "replay";
  const isReplayMode = isSeat1Replay || isSeat2Replay;

  // When one seat becomes replay, force the other to be replay too
  useEffect(() => {
    if (isSeat1Replay && !isSeat2Replay) {
      setSeat2({ actorType: "agent", agentType: "replay" });
    }
    if (isSeat2Replay && !isSeat1Replay) {
      setSeat1({ actorType: "agent", agentType: "replay" });
    }
  }, [isSeat1Replay, isSeat2Replay]);

  // Auto-load default replay when replay mode is entered
  useEffect(() => {
    if (isReplayMode && !replayData && !replayLoading) {
      loadDefaultReplay();
    }
  }, [isReplayMode]);

  const loadDefaultReplay = async () => {
    setReplayLoading(true);
    setReplayError(null);
    try {
      const resp = await fetch("/replays/full_game_replay.json");
      if (!resp.ok) throw new Error(`Failed to fetch: ${resp.status}`);
      const data: ReplayData = await resp.json();
      if (!data.steps || data.steps.length === 0) {
        throw new Error("Replay file has no steps");
      }
      setReplayData(data);
    } catch (err) {
      setReplayError(err instanceof Error ? err.message : "Failed to load replay");
    } finally {
      setReplayLoading(false);
    }
  };

  const handleActorChange = (seat: 1 | 2, actorType: ActorType) => {
    const setter = seat === 1 ? setSeat1 : setSeat2;
    if (actorType === "human") {
      setter({ actorType: "human", agentType: null });
      // If we're leaving replay mode, clear replay data
      if (isReplayMode) {
        setReplayData(null);
        setReplayError(null);
        // Also reset the other seat if it was replay
        const otherSetter = seat === 1 ? setSeat2 : setSeat1;
        const otherSeat = seat === 1 ? seat2 : seat1;
        if (otherSeat.agentType === "replay") {
          otherSetter({ actorType: "human", agentType: null });
        }
      }
    } else {
      // Default to first non-replay agent type
      const defaultAgent = nonReplayAgents[0]?.type_id ?? null;
      setter({ actorType: "agent", agentType: defaultAgent });
    }
  };

  const handleAgentTypeChange = (seat: 1 | 2, agentType: string) => {
    const setter = seat === 1 ? setSeat1 : setSeat2;
    setter({ actorType: "agent", agentType });

    // If selecting replay, the other seat will be forced to replay by the useEffect
    // If leaving replay, clear replay data
    if (agentType !== "replay" && isReplayMode) {
      setReplayData(null);
      setReplayError(null);
      const otherSetter = seat === 1 ? setSeat2 : setSeat1;
      const otherSeat = seat === 1 ? seat2 : seat1;
      if (otherSeat.agentType === "replay") {
        const defaultAgent = nonReplayAgents[0]?.type_id ?? null;
        otherSetter({ actorType: "agent", agentType: defaultAgent });
      }
    }
  };

  // Validation
  const isConfigValid = (() => {
    // Both seats must have valid config
    for (const s of [seat1, seat2]) {
      if (s.actorType === "agent" && !s.agentType) return false;
    }
    // Replay constraint: if replay, both must be replay and data must be loaded
    if (isReplayMode) {
      if (!isSeat1Replay || !isSeat2Replay) return false;
      if (!replayData) return false;
    }
    return true;
  })();

  const handleStartGame = () => {
    if (!isConfigValid) return;

    const config: GameConfig = {
      seat1: seat1.actorType,
      seat2: seat2.actorType,
    };

    if (seat1.actorType === "agent" && seat1.agentType) {
      config.agent1Type = seat1.agentType;
      if (seat1.agentType === "replay" && replayData) {
        const seat1Actions = replayData.steps
          .filter((s) => s.action.seat === 1)
          .map((s) => convertReplayAction(s.action));
        config.agent1Config = { actions: seat1Actions };
      }
    }
    if (seat2.actorType === "agent" && seat2.agentType) {
      config.agent2Type = seat2.agentType;
      if (seat2.agentType === "replay" && replayData) {
        const seat2Actions = replayData.steps
          .filter((s) => s.action.seat === 2)
          .map((s) => convertReplayAction(s.action));
        config.agent2Config = { actions: seat2Actions };
      }
    }

    onStart(config);
  };

  const renderSeatPanel = (seat: 1 | 2) => {
    const seatConfig = seat === 1 ? seat1 : seat2;
    const playerLabel = seat === 1 ? "P1" : "P2";
    const color = seat === 1 ? PLAYER_COLORS.P1.primary : PLAYER_COLORS.P2.primary;
    const isOtherReplay = seat === 1 ? isSeat2Replay : isSeat1Replay;

    return (
      <div
        style={{
          padding: "1rem 1.5rem",
          border: `3px solid ${color}`,
          borderRadius: "10px",
          backgroundColor: "#fafafa",
          minWidth: "280px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1rem" }}>
          <div
            style={{
              backgroundColor: color,
              color: "white",
              padding: "4px 12px",
              borderRadius: "6px",
              fontWeight: "bold",
              fontSize: "1.1rem",
            }}
          >
            {playerLabel}
          </div>
          <span style={{ fontWeight: "bold", fontSize: "1.1rem" }}>Seat {seat}</span>
        </div>

        {/* Actor type selector */}
        <div style={{ marginBottom: "0.75rem" }}>
          <label style={{ fontWeight: "bold", display: "block", marginBottom: "0.25rem" }}>
            Player Type
          </label>
          <select
            value={seatConfig.actorType}
            onChange={(e) => handleActorChange(seat, e.target.value as ActorType)}
            style={selectStyle}
          >
            <option value="human">Human</option>
            <option value="agent">Agent</option>
          </select>
        </div>

        {/* Agent type selector (only when agent) */}
        {seatConfig.actorType === "agent" && (
          <div>
            <label style={{ fontWeight: "bold", display: "block", marginBottom: "0.25rem" }}>
              Agent Type
            </label>
            <select
              value={seatConfig.agentType ?? ""}
              onChange={(e) => handleAgentTypeChange(seat, e.target.value)}
              disabled={isOtherReplay}
              style={selectStyle}
            >
              {nonReplayAgents.map((a) => (
                <option key={a.type_id} value={a.type_id}>
                  {a.display_name}
                </option>
              ))}
              {hasReplayAgent && (
                <option value="replay">Replay Agent</option>
              )}
            </select>
            {isOtherReplay && (
              <div style={{ fontSize: "0.8rem", color: "#888", marginTop: "0.25rem" }}>
                Locked to replay (both seats must match)
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "1.5rem",
        paddingTop: "2rem",
        paddingBottom: "2rem",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <h1 style={{ fontSize: "2.5rem", color: "#333", margin: 0 }}>
        Quoridor
      </h1>
      <h2 style={{ fontSize: "1.2rem", color: "#666", margin: 0, fontWeight: "normal" }}>
        Game Setup
      </h2>

      {/* Seat panels */}
      <div style={{ display: "flex", gap: "2rem", flexWrap: "wrap", justifyContent: "center" }}>
        {renderSeatPanel(1)}
        {renderSeatPanel(2)}
      </div>

      {/* Replay configuration */}
      {isReplayMode && (
        <div
          style={{
            padding: "1rem 1.5rem",
            border: "2px solid #999",
            borderRadius: "10px",
            backgroundColor: "#f0f0f0",
            minWidth: "280px",
            maxWidth: "600px",
            textAlign: "center",
          }}
        >
          <div style={{ fontWeight: "bold", marginBottom: "0.5rem" }}>Replay Data</div>
          {replayLoading && <div style={{ color: "#666" }}>Loading default replay...</div>}
          {replayError && (
            <div>
              <div style={{ color: "#dc3545", marginBottom: "0.5rem" }}>{replayError}</div>
              <button onClick={loadDefaultReplay} style={smallButtonStyle}>
                Retry
              </button>
            </div>
          )}
          {replayData && (
            <div style={{ color: "#28a745" }}>
              Default replay loaded ({replayData.steps.length} steps)
            </div>
          )}
        </div>
      )}

      {/* Error display */}
      {error && (
        <div style={{ color: "#dc3545", fontWeight: "bold" }}>{error}</div>
      )}

      {/* Action buttons */}
      <div style={{ display: "flex", gap: "1rem", marginTop: "0.5rem" }}>
        <button onClick={onBack} style={backButtonStyle}>
          ← Back
        </button>
        <button
          onClick={handleStartGame}
          disabled={!isConfigValid}
          style={{
            ...startButtonStyle,
            opacity: isConfigValid ? 1 : 0.5,
            cursor: isConfigValid ? "pointer" : "not-allowed",
          }}
        >
          Start Game
        </button>
      </div>
    </div>
  );
};

const selectStyle: React.CSSProperties = {
  width: "100%",
  padding: "0.5rem",
  fontSize: "1rem",
  border: "2px solid #ccc",
  borderRadius: "6px",
  backgroundColor: "#fff",
};

const startButtonStyle: React.CSSProperties = {
  padding: "0.75rem 2rem",
  fontSize: "1.1rem",
  fontWeight: "bold",
  border: "none",
  borderRadius: "8px",
  background: "#28a745",
  color: "#fff",
  cursor: "pointer",
};

const backButtonStyle: React.CSSProperties = {
  padding: "0.75rem 1.5rem",
  fontSize: "1rem",
  border: "2px solid #666",
  borderRadius: "8px",
  background: "#fff",
  color: "#333",
  cursor: "pointer",
};

const smallButtonStyle: React.CSSProperties = {
  padding: "0.4rem 1rem",
  fontSize: "0.9rem",
  border: "1px solid #999",
  borderRadius: "6px",
  background: "#fff",
  cursor: "pointer",
};

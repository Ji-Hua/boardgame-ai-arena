// ReplayPage.tsx
//
// Page for replaying a game through the backend.
// Uses ReplayController which sends actions via LiveController → WebSocket → Backend.
// UI updates come from backend state_update — same pipeline as live game.

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Board } from "../../components/board/Board";
import type { PawnData } from "../../components/board/Pawn";
import type { WallData } from "../../components/board/Wall";
import { WallStock } from "../../components/board/WallStock";
import { VictoryModal } from "../../components/UI/VictoryModal";
import { ReplayController } from "../../modes/replay/ReplayController";
import { renderStore } from "../../stores/renderStore";
import type { RenderState } from "../../core/RenderState";
import type { ReplayData } from "../../types/Replay";
import { PLAYER_COLORS } from "../../theme/playerColors";
import "../pages/GamePage.css";

interface ReplayPageProps {
  replayData: ReplayData;
  onBack: () => void;
}

export const ReplayPage: React.FC<ReplayPageProps> = ({
  replayData,
  onBack,
}) => {
  const controllerRef = useRef<ReplayController | null>(null);
  const [renderState, setRenderState] = useState<RenderState>(
    renderStore.getState(),
  );
  const [, setTick] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Stable reference to controller
  const getController = useCallback(() => controllerRef.current, []);

  // Initialize: bootstrap backend game
  useEffect(() => {
    const ctrl = new ReplayController(replayData);
    controllerRef.current = ctrl;

    const unsub = renderStore.subscribe((state) => setRenderState(state));
    const unsubCtrl = ctrl.subscribe(() => setTick((t) => t + 1));

    ctrl.init().then(() => {
      setReady(true);
      setPlaying(true); // auto-play after init
    }).catch((err) => {
      setInitError(err instanceof Error ? err.message : "Failed to initialize replay");
    });

    return () => {
      unsub();
      unsubCtrl();
      ctrl.disconnect();
      controllerRef.current = null;
    };
  }, [replayData]);

  // Auto-play: step forward every 500ms
  useEffect(() => {
    const ctrl = getController();
    if (!ctrl || !ready) return;

    if (playing && !ctrl.isAtEnd && !ctrl.isBusy) {
      timerRef.current = setTimeout(async () => {
        if (ctrl.isAtEnd) {
          setPlaying(false);
          return;
        }
        await ctrl.stepForward();
        setTick((t) => t + 1);
        // Check if we reached the end
        if (ctrl.isAtEnd || ctrl.gameEnded) {
          setPlaying(false);
        }
      }, 500);
    }
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [playing, ready, getController, renderState]); // renderState dep triggers re-schedule after each update

  const togglePlay = useCallback(async () => {
    const ctrl = getController();
    if (!ctrl) return;
    if (ctrl.isAtEnd) {
      setPlaying(false);
      await ctrl.jumpToStart();
      setPlaying(true);
    } else {
      setPlaying((p) => !p);
    }
  }, [getController]);

  // Keyboard navigation
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const ctrl = getController();
      if (!ctrl || !ready || ctrl.isBusy) return;

      if (e.key === "ArrowRight") {
        e.preventDefault();
        setPlaying(false);
        ctrl.stepForward();
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        setPlaying(false);
        ctrl.stepBackward();
      } else if (e.key === "Home") {
        e.preventDefault();
        setPlaying(false);
        ctrl.jumpToStart();
      } else if (e.key === "End") {
        e.preventDefault();
        setPlaying(false);
        ctrl.jumpToEnd();
      } else if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [getController, ready, togglePlay]);

  const ctrl = getController();
  const stepIndex = ctrl?.stepIndex ?? -1;
  const currentStep = ctrl?.currentStep ?? null;
  const totalSteps = ctrl?.totalSteps ?? 0;
  const isAtStart = ctrl?.isAtStart ?? true;
  const isAtEnd = ctrl?.isAtEnd ?? false;

  const seatToPlayer = (s: 1 | 2): "P1" | "P2" => (s === 1 ? "P1" : "P2");

  const pawns: PawnData[] = [
    {
      playerId: "P1",
      row: renderState.pawns[1][0],
      col: renderState.pawns[1][1],
      color: PLAYER_COLORS.P1.primary,
    },
    {
      playerId: "P2",
      row: renderState.pawns[2][0],
      col: renderState.pawns[2][1],
      color: PLAYER_COLORS.P2.primary,
    },
  ];

  const walls: WallData[] = renderState.walls.map((wall) => ({
    orientation: wall.orientation === "H" ? "H" : "V",
    row: wall.row,
    col: wall.col,
    owner: wall.owner ? seatToPlayer(wall.owner) : undefined,
    color: wall.owner
      ? PLAYER_COLORS[seatToPlayer(wall.owner)].primary
      : PLAYER_COLORS.P1.primary,
  }));

  const stepLabel =
    stepIndex < 0
      ? "Initial State"
      : `Step ${currentStep?.stepId ?? stepIndex + 1}`;
  const outcomeLabel =
    currentStep?.outcome === "accept"
      ? "✓ ACCEPT"
      : currentStep?.outcome === "reject"
        ? "✗ REJECT"
        : "";
  const outcomeColor =
    currentStep?.outcome === "accept" ? "#28a745" : "#dc3545";

  if (initError) {
    return (
      <div className="game-page">
        <div style={{ textAlign: "center", paddingTop: "4rem" }}>
          <h1 style={{ fontSize: "2rem", color: "#333" }}>Replay Error</h1>
          <p style={{ color: "#dc3545" }}>{initError}</p>
          <button onClick={onBack} style={{ padding: "0.5rem 1rem", cursor: "pointer" }}>
            ← Back
          </button>
        </div>
      </div>
    );
  }

  if (!ready) {
    return (
      <div className="game-page">
        <div style={{ textAlign: "center", paddingTop: "4rem" }}>
          <h1 style={{ fontSize: "2rem", color: "#333" }}>Starting replay...</h1>
        </div>
      </div>
    );
  }

  return (
    <div className="game-page">
      <header className="game-header">
        <h1 className="game-title">Quoridor — Replay</h1>
        <button
          onClick={onBack}
          style={{
            padding: "0.4rem 1rem",
            background: "#666",
            color: "white",
            border: "none",
            borderRadius: "6px",
            cursor: "pointer",
          }}
        >
          ← Back
        </button>
      </header>

      <div className="game-content">
        {/* Player 2 info (top) */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "1rem",
            padding: "0.75rem 1rem",
            backgroundColor:
              renderState.currentSeat === 2 ? "#fff" : "#f5f5f5",
            border: `3px solid ${PLAYER_COLORS.P2.primary}`,
            borderRadius: "8px",
          }}
        >
          <div
            style={{
              backgroundColor: PLAYER_COLORS.P2.primary,
              color: "white",
              padding: "4px 10px",
              borderRadius: "6px",
              fontWeight: "bold",
            }}
          >
            P2
          </div>
          <span style={{ fontWeight: "bold" }}>Player 2</span>
          <span style={{ color: "#666" }}>
            Walls: {renderState.wallsRemaining[2]}
          </span>
        </div>

        {/* Wall Stock P2 */}
        <WallStock
          player="P2"
          wallsRemaining={renderState.wallsRemaining[2]}
          orientation="horizontal"
          onEnterWallMode={() => {}}
          isActive={false}
        />

        {/* Board */}
        <div className="board-section">
          <Board
            boardSize={renderState.boardSize}
            cellSize={48}
            pawns={pawns}
            walls={walls}
            mode="replay"
            currentPlayer={renderState.currentSeat === 1 ? "P1" : "P2"}
          />
        </div>

        {/* Wall Stock P1 */}
        <WallStock
          player="P1"
          wallsRemaining={renderState.wallsRemaining[1]}
          orientation="horizontal"
          onEnterWallMode={() => {}}
          isActive={false}
        />

        {/* Player 1 info (bottom) */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "1rem",
            padding: "0.75rem 1rem",
            backgroundColor:
              renderState.currentSeat === 1 ? "#fff" : "#f5f5f5",
            border: `3px solid ${PLAYER_COLORS.P1.primary}`,
            borderRadius: "8px",
          }}
        >
          <div
            style={{
              backgroundColor: PLAYER_COLORS.P1.primary,
              color: "white",
              padding: "4px 10px",
              borderRadius: "6px",
              fontWeight: "bold",
            }}
          >
            P1
          </div>
          <span style={{ fontWeight: "bold" }}>Player 1</span>
          <span style={{ color: "#666" }}>
            Walls: {renderState.wallsRemaining[1]}
          </span>
        </div>
      </div>

      {/* Replay Controls */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "0.5rem",
          padding: "1rem",
          borderTop: "1px solid #e0e0e0",
          background: "#fafafa",
        }}
      >
        {/* Step info */}
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          <span style={{ fontWeight: "bold", fontSize: "1.1rem" }}>
            {stepLabel}
          </span>
          {currentStep && (
            <span
              style={{
                fontWeight: "bold",
                color: outcomeColor,
                fontSize: "0.95rem",
              }}
            >
              {outcomeLabel}
            </span>
          )}
          <span style={{ color: "#888", fontSize: "0.9rem" }}>
            ({stepIndex + 1} / {totalSteps})
          </span>
        </div>

        {/* Action info */}
        {currentStep && (
          <div style={{ color: "#555", fontSize: "0.85rem" }}>
            Player {currentStep.action.seat} —{" "}
            {currentStep.action.type === "pawn"
              ? `Move to [${currentStep.action.position[0]}, ${currentStep.action.position[1]}]`
              : `Place ${currentStep.action.type} wall at [${currentStep.action.position[0]}, ${currentStep.action.position[1]}]`}
          </div>
        )}

        {/* Navigation buttons */}
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button
            onClick={() => { setPlaying(false); ctrl?.jumpToStart(); }}
            disabled={isAtStart || (ctrl?.isBusy ?? false)}
            style={navButtonStyle(isAtStart || (ctrl?.isBusy ?? false))}
          >
            ⏮ Start
          </button>
          <button
            onClick={() => { setPlaying(false); ctrl?.stepBackward(); }}
            disabled={isAtStart || (ctrl?.isBusy ?? false)}
            style={navButtonStyle(isAtStart || (ctrl?.isBusy ?? false))}
          >
            ◀ Prev
          </button>
          <button
            onClick={togglePlay}
            disabled={ctrl?.isBusy ?? false}
            style={{
              padding: "0.5rem 1.2rem",
              fontSize: "0.9rem",
              border: "2px solid #333",
              borderRadius: "6px",
              background: playing ? "#333" : "#fff",
              color: playing ? "#fff" : "#333",
              cursor: (ctrl?.isBusy ?? false) ? "not-allowed" : "pointer",
              fontWeight: "bold",
            }}
          >
            {playing ? "⏸ Pause" : isAtEnd ? "⏮ Restart" : "▶ Play"}
          </button>
          <button
            onClick={() => { setPlaying(false); ctrl?.stepForward(); }}
            disabled={isAtEnd || (ctrl?.isBusy ?? false)}
            style={navButtonStyle(isAtEnd || (ctrl?.isBusy ?? false))}
          >
            Next ▶
          </button>
          <button
            onClick={() => { setPlaying(false); ctrl?.jumpToEnd(); }}
            disabled={isAtEnd || (ctrl?.isBusy ?? false)}
            style={navButtonStyle(isAtEnd || (ctrl?.isBusy ?? false))}
          >
            End ⏭
          </button>
        </div>

        <div style={{ color: "#999", fontSize: "0.75rem" }}>
          Space = play/pause · ← → = step · Home/End = jump
        </div>
      </div>

      {/* Victory Modal at end */}
      {renderState.isTerminal && renderState.result && (
        <VictoryModal
          winner={renderState.result.winnerSeat}
          termination={renderState.result.termination}
          onNewGame={() => { ctrl?.jumpToStart(); }}
          onNewMatch={onBack}
        />
      )}
    </div>
  );
};

function navButtonStyle(
  disabled: boolean,
): React.CSSProperties {
  return {
    padding: "0.5rem 1rem",
    fontSize: "0.9rem",
    border: "1px solid #ccc",
    borderRadius: "6px",
    background: disabled ? "#f0f0f0" : "#fff",
    color: disabled ? "#bbb" : "#333",
    cursor: disabled ? "not-allowed" : "pointer",
    fontWeight: "bold",
  };
}

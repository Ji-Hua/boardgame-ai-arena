// GamePage.tsx
// Main page for local-mode game.
// Auto-bootstraps on mount: creates room, joins both seats, starts game.
// Single client controls both seats via WebSocket.

import React, { useState, useEffect, useRef } from "react";
import {
  Board,
  WallPlacement,
  WallPreview,
} from "../../components/board/Board";
import type { PawnData } from "../../components/board/Pawn";
import type { WallData } from "../../components/board/Wall";
import { VictoryModal } from "../../components/UI/VictoryModal";
import { MatchControls } from "../../components/UI/MatchControls";
import { WallStock } from "../../components/board/WallStock";
import { LiveController } from "../../modes/live/LiveController";
import { renderStore } from "../../stores/renderStore";
import type { RenderState } from "../../core/RenderState";
import { PLAYER_COLORS } from "../../theme/playerColors";
import type { ActionWire } from "../../api/websocket";
import "./GamePage.css";

type GameFlow = "loading" | "playing" | "ended" | "error";

interface GamePageProps {
  onBack?: () => void;
}

export const GamePage: React.FC<GamePageProps> = ({ onBack }) => {
  const [controller] = useState(() => new LiveController());
  const [renderState, setRenderState] = useState<RenderState>(
    renderStore.getState(),
  );
  const [flow, setFlow] = useState<GameFlow>("loading");
  const [error, setError] = useState<string | null>(null);

  // Board interaction state
  const [wallPlacement, setWallPlacement] = useState<WallPlacement>({
    isActive: false,
    playerId: null,
  });
  const [wallPreview, setWallPreview] = useState<WallPreview | null>(null);
  const [wallError, setWallError] = useState<string | null>(null);
  const [legalMoves, setLegalMoves] = useState<[number, number][]>([]);
  const [, setTick] = useState(0);
  const initializedRef = useRef(false);

  // ── Auto-bootstrap on mount ──────────────────────────────────

  useEffect(() => {
    if (initializedRef.current) return;
    initializedRef.current = true;

    const init = async () => {
      try {
        await controller.bootstrap();
        setFlow("playing");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to initialize");
        setFlow("error");
      }
    };
    init();

    return () => controller.disconnect();
  }, [controller]);

  // ── Subscriptions ────────────────────────────────────────────

  useEffect(() => {
    const unsub = renderStore.subscribe((state) => {
      setRenderState(state);
      setLegalMoves([]);  // Clear highlights on any state change
      if (state.isTerminal && flow !== "ended") {
        setFlow("ended");
      }
    });
    return unsub;
  }, [controller, flow]);

  useEffect(() => {
    const unsub = controller.subscribe(() => {
      setTick((t) => t + 1);
      if (controller.gameEnded) setFlow("ended");
      if (controller.lastError) {
        setError(controller.lastError);
        controller.lastError = null;
      }
    });
    return unsub;
  }, [controller]);

  // ── Action handlers ──────────────────────────────────────────

  const handleMovePawn = (row: number, col: number) => {
    // row = engine_y, col = engine_x; backend expects target: [x, y]
    const action: ActionWire = {
      player: renderState.currentSeat,
      type: "pawn",
      target: [col, row],
    };
    controller.takeAction(action);
    setLegalMoves([]);  // Clear highlights after move
  };

  const handlePawnClick = async () => {
    // Toggle: if highlights already showing, clear them; otherwise fetch from backend.
    if (legalMoves.length > 0) {
      setLegalMoves([]);
      return;
    }
    try {
      const moves = await controller.getLegalPawnMoves();
      setLegalMoves(moves);
    } catch {
      setLegalMoves([]);
    }
  };

  const handleConfirmWall = (preview: WallPreview) => {
    // preview.logicalRow = engine_y, preview.logicalCol = engine_x
    // backend expects target: [x, y]
    const action: ActionWire = {
      player: renderState.currentSeat,
      type: preview.orientation === "Horizontal" ? "horizontal" : "vertical",
      target: [preview.logicalCol, preview.logicalRow],
    };
    controller.takeAction(action);
    setWallPlacement({ isActive: false, playerId: null });
    setWallPreview(null);
  };

  const handleEnterWallMode = (player: "P1" | "P2") => {
    const seat = player === "P1" ? 1 : 2;
    if (
      renderState.wallsRemaining[seat as 1 | 2] > 0 &&
      renderState.currentSeat === seat &&
      !renderState.isTerminal
    ) {
      setLegalMoves([]);  // Clear pawn highlights when entering wall mode
      setWallPlacement({ isActive: true, playerId: player });
    }
  };

  const handleCancelWallPlacement = () => {
    setWallPlacement({ isActive: false, playerId: null });
    setWallPreview(null);
  };

  const handleSurrender = () => {
    controller.surrender(renderState.currentSeat);
  };

  const handleNewMatch = async () => {
    controller.disconnect();
    initializedRef.current = false;
    setWallPlacement({ isActive: false, playerId: null });
    setWallPreview(null);
    setWallError(null);
    setLegalMoves([]);
    setError(null);
    setFlow("loading");

    // Re-bootstrap
    try {
      await controller.bootstrap();
      setFlow("playing");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to restart");
      setFlow("error");
    }
  };

  // ── Loading / Error states ───────────────────────────────────

  if (flow === "loading") {
    return (
      <div className="game-page">
        <div style={{ textAlign: "center", paddingTop: "4rem" }}>
          <h1 style={{ fontSize: "2rem", color: "#333" }}>Connecting...</h1>
        </div>
      </div>
    );
  }

  if (flow === "error") {
    return (
      <div className="game-page">
        <div style={{ textAlign: "center", paddingTop: "4rem" }}>
          <h1 style={{ fontSize: "2rem", color: "#333" }}>Error</h1>
          <p style={{ color: "#dc3545" }}>{error}</p>
          <button className="new-game-button" onClick={handleNewMatch}>
            Try Again
          </button>
        </div>
      </div>
    );
  }

  // ── Board play (playing / ended) ─────────────────────────────

  const isPlaying = flow === "playing";
  const isEnded = flow === "ended" || renderState.isTerminal;

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

  const seatToPlayer = (s: 1 | 2): "P1" | "P2" => (s === 1 ? "P1" : "P2");

  const walls: WallData[] = renderState.walls.map((wall) => ({
    orientation: wall.orientation === "H" ? "H" : "V",
    row: wall.row,
    col: wall.col,
    owner: wall.owner ? seatToPlayer(wall.owner) : undefined,
    color: wall.owner
      ? PLAYER_COLORS[seatToPlayer(wall.owner)].primary
      : PLAYER_COLORS.P1.primary,
  }));

  return (
    <div className="game-page">
      <header className="game-header">
        <h1 className="game-title">Quoridor</h1>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <MatchControls
            isPreGame={false}
            isInGame={isPlaying || isEnded}
            onNewGame={handleNewMatch}
            onNewMatch={handleNewMatch}
          />
          {onBack && (
            <button
              onClick={() => { controller.disconnect(); onBack(); }}
              style={{
                padding: "0.4rem 1rem",
                background: "#666",
                color: "white",
                border: "none",
                borderRadius: "6px",
                cursor: "pointer",
              }}
            >
              ← Lobby
            </button>
          )}
        </div>
      </header>

      <div className="game-content">
        {/* Player 2 info (top) */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          padding: "0.75rem 1rem",
          backgroundColor: renderState.currentSeat === 2 && !isEnded ? "#fff" : "#f5f5f5",
          border: `3px solid ${PLAYER_COLORS.P2.primary}`,
          borderRadius: "8px",
          boxShadow: renderState.currentSeat === 2 && !isEnded ? `0 0 20px ${PLAYER_COLORS.P2.primary}40` : "none",
        }}>
          {renderState.currentSeat === 2 && !isEnded && (
            <span style={{ color: PLAYER_COLORS.P2.primary }}>▶</span>
          )}
          <div style={{ backgroundColor: PLAYER_COLORS.P2.primary, color: "white", padding: "4px 10px", borderRadius: "6px", fontWeight: "bold" }}>P2</div>
          <span style={{ fontWeight: "bold" }}>Player 2</span>
          <span style={{ color: "#666" }}>Walls: {renderState.wallsRemaining[2]}</span>
          {!isEnded && (
            <button
              onClick={handleSurrender}
              disabled={renderState.currentSeat !== 2}
              style={{
                marginLeft: "auto",
                padding: "0.4rem 0.8rem",
                backgroundColor: renderState.currentSeat === 2 ? "#dc3545" : "#e0e0e0",
                color: renderState.currentSeat === 2 ? "white" : "#999",
                border: "none",
                borderRadius: "6px",
                cursor: renderState.currentSeat === 2 ? "pointer" : "not-allowed",
                opacity: renderState.currentSeat === 2 ? 1 : 0.5,
              }}
            >
              🏳️ Surrender
            </button>
          )}
        </div>

        {/* Wall Stock P2 */}
        <WallStock
          player="P2"
          wallsRemaining={renderState.wallsRemaining[2]}
          orientation="horizontal"
          onEnterWallMode={() => handleEnterWallMode("P2")}
          isActive={
            isPlaying &&
            wallPlacement.isActive &&
            wallPlacement.playerId === "P2"
          }
        />

        {/* Board */}
        <div
          className="board-section"
          style={{
            position: "relative",
            pointerEvents: isEnded ? "none" : "auto",
            opacity: 1,
          }}
        >
          <Board
            boardSize={renderState.boardSize}
            cellSize={48}
            pawns={pawns}
            walls={walls}
            mode="play"
            legalMoves={legalMoves}
            currentPlayer={renderState.currentSeat === 1 ? "P1" : "P2"}
            wallPlacement={wallPlacement}
            wallPreview={wallPreview}
            onUpdateWallPreview={setWallPreview}
            onConfirmWall={handleConfirmWall}
            onMovePawn={handleMovePawn}
            onPawnClick={handlePawnClick}
            onCancelWallPlacement={handleCancelWallPlacement}
            externalWallError={wallError}
            onClearExternalWallError={() => setWallError(null)}
          />
        </div>

        {/* Wall Stock P1 */}
        <WallStock
          player="P1"
          wallsRemaining={renderState.wallsRemaining[1]}
          orientation="horizontal"
          onEnterWallMode={() => handleEnterWallMode("P1")}
          isActive={
            isPlaying &&
            wallPlacement.isActive &&
            wallPlacement.playerId === "P1"
          }
        />

        {/* Player 1 info (bottom) */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          padding: "0.75rem 1rem",
          backgroundColor: renderState.currentSeat === 1 && !isEnded ? "#fff" : "#f5f5f5",
          border: `3px solid ${PLAYER_COLORS.P1.primary}`,
          borderRadius: "8px",
          boxShadow: renderState.currentSeat === 1 && !isEnded ? `0 0 20px ${PLAYER_COLORS.P1.primary}40` : "none",
        }}>
          {renderState.currentSeat === 1 && !isEnded && (
            <span style={{ color: PLAYER_COLORS.P1.primary }}>▶</span>
          )}
          <div style={{ backgroundColor: PLAYER_COLORS.P1.primary, color: "white", padding: "4px 10px", borderRadius: "6px", fontWeight: "bold" }}>P1</div>
          <span style={{ fontWeight: "bold" }}>Player 1</span>
          <span style={{ color: "#666" }}>Walls: {renderState.wallsRemaining[1]}</span>
          {!isEnded && (
            <button
              onClick={handleSurrender}
              disabled={renderState.currentSeat !== 1}
              style={{
                marginLeft: "auto",
                padding: "0.4rem 0.8rem",
                backgroundColor: renderState.currentSeat === 1 ? "#dc3545" : "#e0e0e0",
                color: renderState.currentSeat === 1 ? "white" : "#999",
                border: "none",
                borderRadius: "6px",
                cursor: renderState.currentSeat === 1 ? "pointer" : "not-allowed",
                opacity: renderState.currentSeat === 1 ? 1 : 0.5,
              }}
            >
              🏳️ Surrender
            </button>
          )}
        </div>
      </div>

      {error && flow === "playing" && (
        <div
          style={{
            position: "fixed",
            bottom: "1rem",
            left: "50%",
            transform: "translateX(-50%)",
            background: "#dc3545",
            color: "white",
            padding: "0.75rem 1.5rem",
            borderRadius: "8px",
            zIndex: 2000,
            cursor: "pointer",
          }}
          onClick={() => setError(null)}
        >
          {error}
        </div>
      )}

      {/* Victory Modal */}
      {renderState.isTerminal && renderState.result && (
        <VictoryModal
          winner={renderState.result.winnerSeat}
          termination={renderState.result.termination}
          onNewGame={handleNewMatch}
          onNewMatch={handleNewMatch}
        />
      )}
    </div>
  );
};

import React from "react";

interface VictoryModalProps {
  winner: number | null;
  termination: string | null;
  onNewGame: () => void;
  onNewMatch: () => void;
}

export const VictoryModal: React.FC<VictoryModalProps> = ({ winner, termination, onNewGame, onNewMatch }) => {
  // winner is a seat (1 or 2), display as P1/P2 for UI clarity
  const winnerLabel = winner ? `P${winner}` : null;
  const title = winnerLabel ? `${winnerLabel} wins!` : "Game Over";
  const subtitle = termination === "surrender" && winnerLabel ? `${winnerLabel} wins by surrender` : undefined;

  return (
    <div className="victory-backdrop" role="dialog" aria-modal="true">
      <div className="victory-modal">
        <h1 style={{ margin: 0, fontSize: "1.5rem" }}>{title}</h1>
        {subtitle && <p style={{ marginTop: "0.75rem", color: "#555" }}>{subtitle}</p>}
        <div style={{ marginTop: "1.25rem", display: "flex", gap: "0.75rem", justifyContent: "center" }}>
          <button onClick={onNewGame}>
            New Game
          </button>
          <button onClick={onNewMatch} style={{ background: "#666" }}>
            New Match
          </button>
        </div>
      </div>
      <style>{`
        .victory-backdrop {
          position: fixed;
          top: 0; left: 0;
          width: 100%; height: 100%;
          background: rgba(0,0,0,0.5);
          display: flex; align-items: center; justify-content: center;
          z-index: 2000;
        }
        .victory-modal {
          background: white;
          padding: 2rem;
          border-radius: 10px;
          text-align: center;
          width: 300px;
          box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .victory-modal button {
          background: #333;
          color: white;
          border: none;
          border-radius: 6px;
          padding: 0.5rem 1rem;
          cursor: pointer;
        }
        .victory-modal button:hover {
          opacity: 0.9;
        }
      `}</style>
    </div>
  );
};

import { useState } from "react";
import { GamePage } from "./view/pages/GamePage";
import { ReplayPage } from "./view/replay/ReplayPage";
import type { ReplayData } from "./types/Replay";
import "./App.css";

type AppMode = "menu" | "live" | "replay";

function App() {
  const [mode, setMode] = useState<AppMode>("menu");
  const [replayData, setReplayData] = useState<ReplayData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleLoadReplay = async () => {
    try {
      setError(null);
      const resp = await fetch("/replays/full_game_replay.json");
      if (!resp.ok) throw new Error(`Failed to fetch replay: ${resp.status}`);
      const data: ReplayData = await resp.json();
      if (data.steps.length === 0) {
        throw new Error("No steps in replay file");
      }
      setReplayData(data);
      setMode("replay");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load replay");
    }
  };

  if (mode === "live") {
    return <GamePage onBack={() => setMode("menu")} />;
  }

  if (mode === "replay" && replayData) {
    return <ReplayPage replayData={replayData} onBack={() => setMode("menu")} />;
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "2rem",
        paddingTop: "4rem",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <h1 style={{ fontSize: "2.5rem", color: "#333", margin: 0 }}>
        Quoridor
      </h1>
      <div style={{ display: "flex", gap: "1.5rem" }}>
        <button onClick={() => setMode("live")} style={menuButtonStyle}>
          ▶ Play Local Game
        </button>
        <button onClick={handleLoadReplay} style={menuButtonStyle}>
          📼 Replay Full Game
        </button>
      </div>
      {error && (
        <div style={{ color: "#dc3545", fontWeight: "bold" }}>{error}</div>
      )}
    </div>
  );
}

const menuButtonStyle: React.CSSProperties = {
  padding: "1rem 2rem",
  fontSize: "1.1rem",
  fontWeight: "bold",
  border: "2px solid #333",
  borderRadius: "8px",
  background: "#fff",
  color: "#333",
  cursor: "pointer",
};

export default App;

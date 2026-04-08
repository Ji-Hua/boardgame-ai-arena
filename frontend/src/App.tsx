import { useState } from "react";
import { GamePage } from "./view/pages/GamePage";
import { ConfigPage } from "./view/pages/ConfigPage";
import type { GameConfig } from "./modes/live/LiveController";
import "./App.css";

type AppMode = "menu" | "config" | "game";

function App() {
  const [mode, setMode] = useState<AppMode>("menu");
  const [gameConfig, setGameConfig] = useState<GameConfig>({ seat1: "human", seat2: "human" });

  const handleStartGame = (config: GameConfig) => {
    setGameConfig(config);
    setMode("game");
  };

  if (mode === "config") {
    return <ConfigPage onStart={handleStartGame} onBack={() => setMode("menu")} />;
  }

  if (mode === "game") {
    return <GamePage gameConfig={gameConfig} onBack={() => setMode("config")} />;
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
      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <button
          onClick={() => setMode("config")}
          style={menuButtonStyle}
        >
          Play
        </button>
      </div>
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

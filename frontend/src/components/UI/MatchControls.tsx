import React from 'react';

interface MatchControlsProps {
  // Game state
  isPreGame: boolean;
  isInGame: boolean;
  
  // Control handlers
  onSwitchSeats?: () => void;
  onNewGame?: () => void;
  onNewMatch?: () => void;
  
  // Control visibility overrides
  showSwitchSeats?: boolean;
  showNewGame?: boolean;
  showNewMatch?: boolean;
}

/**
 * MatchControls: Match-level control buttons
 * 
 * Three controls:
 * - Switch Seats: Pre-game only, swaps P1 ↔ P2 via backend
 * - New Game: In-game and game-end, resets board but keeps players
 * - New Match: Always available, full reset to configuration phase
 * 
 * Visibility rules:
 * - Switch Seats: ONLY during pre-game (configuring phase)
 * - New Game: During in-game and game-end
 * - New Match: Always visible
 */
export const MatchControls: React.FC<MatchControlsProps> = ({
  isPreGame,
  isInGame,
  onSwitchSeats,
  onNewGame,
  onNewMatch,
  showSwitchSeats = true,
  showNewGame = true,
  showNewMatch = true,
}) => {
  // Compute visibility based on game state
  // Switch Seats: ONLY Pre-Game (isPreGame=true)
  // New Game: In-Game + Game-End (isPreGame=false)
  // New Match: Always (when handler provided)
  const shouldShowSwitchSeats = isPreGame && showSwitchSeats && onSwitchSeats;
  const shouldShowNewGame = !isPreGame && showNewGame && onNewGame;
  const shouldShowNewMatch = showNewMatch && onNewMatch;

  // Don't render if no controls are visible
  if (!shouldShowSwitchSeats && !shouldShowNewGame && !shouldShowNewMatch) {
    return null;
  }

  return (
    <div style={styles.container}>
      {shouldShowSwitchSeats && (
        <button
          onClick={onSwitchSeats}
          style={{
            ...styles.button,
            ...styles.switchSeatsButton,
          }}
          title="Swap P1 and P2 seat assignments"
        >
          ⇄ Switch Seats
        </button>
      )}
      
      {shouldShowNewGame && (
        <button
          onClick={onNewGame}
          style={{
            ...styles.button,
            ...styles.newGameButton,
          }}
          title="Start a new game with same players"
        >
          🔄 New Game
        </button>
      )}
      
      {shouldShowNewMatch && (
        <button
          onClick={onNewMatch}
          style={{
            ...styles.button,
            ...styles.newMatchButton,
          }}
          title="Reset match and player configuration"
        >
          ↻ New Match
        </button>
      )}
    </div>
  );
};

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    gap: '0.5rem',
    alignItems: 'center',
  },
  button: {
    padding: '0.5rem 1rem',
    fontSize: '0.875rem',
    fontWeight: 'bold',
    color: 'white',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    whiteSpace: 'nowrap',
  },
  switchSeatsButton: {
    backgroundColor: '#6c757d',
  },
  newGameButton: {
    backgroundColor: '#28a745',
  },
  newMatchButton: {
    backgroundColor: '#007bff',
  },
};

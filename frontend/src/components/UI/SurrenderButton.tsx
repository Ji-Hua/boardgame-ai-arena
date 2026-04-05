import React from 'react';

interface SurrenderButtonProps {
  /**
   * Whether it's the current player's turn
   */
  isMyTurn: boolean;

  /**
   * Whether the game is over
   */
  isGameOver: boolean;

  /**
   * Callback when surrender button is clicked
   */
  onSurrender: () => void;
}

/**
 * Surrender button component
 *
 * Allows a player to forfeit the game during their turn.
 * - Only enabled during the current player's turn
 * - Disabled when game is over
 * - Sends surrender action via WebSocket
 */
export const SurrenderButton: React.FC<SurrenderButtonProps> = ({
  isMyTurn,
  isGameOver,
  onSurrender,
}) => {
  const isDisabled = !isMyTurn || isGameOver;

  return (
    <button
      onClick={onSurrender}
      disabled={isDisabled}
      style={{
        ...styles.button,
        ...(isDisabled ? styles.buttonDisabled : styles.buttonEnabled),
      }}
      title={
        isGameOver
          ? 'Game is over'
          : !isMyTurn
          ? 'Wait for your turn'
          : 'Surrender and forfeit the game'
      }
    >
      🏳️ Surrender
    </button>
  );
};

const styles: Record<string, React.CSSProperties> = {
  button: {
    padding: '0.75rem 1.5rem',
    fontSize: '1rem',
    fontWeight: 'bold',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
  },
  buttonEnabled: {
    color: 'white',
    backgroundColor: '#dc3545',
    cursor: 'pointer',
  },
  buttonDisabled: {
    color: '#999',
    backgroundColor: '#e0e0e0',
    cursor: 'not-allowed',
    opacity: 0.6,
  },
};

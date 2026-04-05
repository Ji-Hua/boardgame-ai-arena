import React from 'react';

interface PlayerTurnBadgeProps {
  currentPlayer: 'P1' | 'P2';
  player1WallsRemaining: number;
  player2WallsRemaining: number;
}

/**
 * Displays the current player's turn and walls remaining
 */
export const PlayerTurnBadge: React.FC<PlayerTurnBadgeProps> = ({
  currentPlayer,
  player1WallsRemaining,
  player2WallsRemaining,
}) => {
  const isPlayer1Turn = currentPlayer === 'P1';

  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '1rem',
        backgroundColor: '#f5f5f5',
        borderRadius: '8px',
        marginBottom: '1rem',
      }}
    >
      {/* Player 1 */}
      <div
        style={{
          padding: '0.75rem 1.5rem',
          borderRadius: '8px',
          backgroundColor: isPlayer1Turn ? '#4CAF50' : '#e0e0e0',
          color: isPlayer1Turn ? 'white' : '#666',
          fontWeight: 'bold',
          transition: 'all 0.3s ease',
        }}
      >
        <div style={{ fontSize: '1.2rem', marginBottom: '0.25rem' }}>
          Player 1
        </div>
        <div style={{ fontSize: '0.9rem' }}>
          Walls: {player1WallsRemaining}
        </div>
      </div>

      {/* Turn indicator */}
      <div
        style={{
          fontSize: '1.5rem',
          fontWeight: 'bold',
          color: '#333',
        }}
      >
        {isPlayer1Turn ? '→' : '←'}
      </div>

      {/* Player 2 */}
      <div
        style={{
          padding: '0.75rem 1.5rem',
          borderRadius: '8px',
          backgroundColor: !isPlayer1Turn ? '#2196F3' : '#e0e0e0',
          color: !isPlayer1Turn ? 'white' : '#666',
          fontWeight: 'bold',
          transition: 'all 0.3s ease',
        }}
      >
        <div style={{ fontSize: '1.2rem', marginBottom: '0.25rem' }}>
          Player 2
        </div>
        <div style={{ fontSize: '0.9rem' }}>
          Walls: {player2WallsRemaining}
        </div>
      </div>
    </div>
  );
};

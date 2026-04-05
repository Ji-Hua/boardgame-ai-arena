import React, { useEffect } from 'react';

interface ErrorToastProps {
  message: string | null;
  onClear: () => void;
  autoHideDuration?: number;
}

/**
 * Toast notification for displaying errors
 */
export const ErrorToast: React.FC<ErrorToastProps> = ({
  message,
  onClear,
  autoHideDuration = 5000,
}) => {
  useEffect(() => {
    if (message && autoHideDuration > 0) {
      const timer = setTimeout(() => {
        onClear();
      }, autoHideDuration);

      return () => clearTimeout(timer);
    }
  }, [message, autoHideDuration, onClear]);

  if (!message) return null;

  const isGameOver = message.toLowerCase().includes('game over');
  const isWin = message.toLowerCase().includes('wins');

  return (
    <div
      data-testid="error-toast"
      style={{
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '1rem 1.5rem',
        backgroundColor: isGameOver && isWin ? '#4CAF50' : '#f44336',
        color: 'white',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        zIndex: 1000,
        maxWidth: '400px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1rem',
        animation: 'slideIn 0.3s ease-out',
      }}
    >
      <div style={{ flex: 1 }}>
        {isGameOver && isWin ? (
          <div>
            <div style={{ fontSize: '1.1rem', fontWeight: 'bold' }}>
              🎉 {message}
            </div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: '0.9rem', fontWeight: 'bold' }}>Error</div>
            <div style={{ fontSize: '1rem', marginTop: '0.25rem' }}>
              {message}
            </div>
          </div>
        )}
      </div>

      <button
        onClick={onClear}
        style={{
          background: 'transparent',
          border: 'none',
          color: 'white',
          cursor: 'pointer',
          fontSize: '1.5rem',
          padding: '0',
          lineHeight: '1',
        }}
        aria-label="Close"
      >
        ×
      </button>

      <style>
        {`
          @keyframes slideIn {
            from {
              transform: translateX(100%);
              opacity: 0;
            }
            to {
              transform: translateX(0);
              opacity: 1;
            }
          }
        `}
      </style>
    </div>
  );
};

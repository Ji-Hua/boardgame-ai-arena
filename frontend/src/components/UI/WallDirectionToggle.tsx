import React from 'react';

interface WallDirectionToggleProps {
  orientation: 'H' | 'V';
  onOrientationChange: (orientation: 'H' | 'V') => void;
  wallsRemaining: number;
}

/**
 * Wall direction toggle component
 * Allows user to select horizontal or vertical wall orientation
 */
export const WallDirectionToggle: React.FC<WallDirectionToggleProps> = ({
  orientation,
  onOrientationChange,
  wallsRemaining,
}) => {
  const isHorizontal = orientation === 'H';
  const isVertical = orientation === 'V';

  return (
    <div style={styles.container}>
      <div style={styles.label}>
        <span style={styles.labelText}>Wall Placement</span>
        <span style={styles.wallCount}>
          {wallsRemaining} {wallsRemaining === 1 ? 'wall' : 'walls'} remaining
        </span>
      </div>

      <div style={styles.buttonGroup}>
        <button
          onClick={() => onOrientationChange('H')}
          disabled={wallsRemaining === 0}
          style={{
            ...styles.button,
            ...(isHorizontal ? styles.buttonActive : {}),
            ...(wallsRemaining === 0 ? styles.buttonDisabled : {}),
          }}
          title="Horizontal wall (spans 2 cells horizontally)"
        >
          <div style={styles.wallPreview}>
            <div style={styles.wallHorizontal}></div>
          </div>
          <span>Horizontal</span>
        </button>

        <button
          onClick={() => onOrientationChange('V')}
          disabled={wallsRemaining === 0}
          style={{
            ...styles.button,
            ...(isVertical ? styles.buttonActive : {}),
            ...(wallsRemaining === 0 ? styles.buttonDisabled : {}),
          }}
          title="Vertical wall (spans 2 cells vertically)"
        >
          <div style={styles.wallPreview}>
            <div style={styles.wallVertical}></div>
          </div>
          <span>Vertical</span>
        </button>
      </div>

      <div style={styles.hint}>
        {wallsRemaining > 0 ? (
          <>
            <span style={styles.hintIcon}>💡</span>
            Click between cells on the board to place a {orientation === 'H' ? 'horizontal' : 'vertical'} wall
          </>
        ) : (
          <>
            <span style={styles.hintIcon}>⚠️</span>
            No walls remaining
          </>
        )}
      </div>
    </div>
  );
};

// Styles
const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
    padding: '1rem',
    backgroundColor: 'white',
    borderRadius: '8px',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
    marginBottom: '1rem',
    minWidth: '300px',
  },
  label: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  labelText: {
    fontSize: '1rem',
    fontWeight: 'bold',
    color: '#333',
  },
  wallCount: {
    fontSize: '0.875rem',
    color: '#666',
    fontWeight: 'normal',
  },
  buttonGroup: {
    display: 'flex',
    gap: '0.5rem',
  },
  button: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '1rem',
    backgroundColor: '#f5f5f5',
    border: '2px solid #ddd',
    borderRadius: '8px',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
    fontSize: '0.875rem',
    fontWeight: '500',
    color: '#666',
  },
  buttonActive: {
    backgroundColor: '#e3f2fd',
    borderColor: '#2196F3',
    color: '#2196F3',
    fontWeight: 'bold',
  },
  buttonDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  wallPreview: {
    width: '40px',
    height: '40px',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
  },
  wallHorizontal: {
    width: '36px',
    height: '6px',
    backgroundColor: '#333',
    borderRadius: '3px',
  },
  wallVertical: {
    width: '6px',
    height: '36px',
    backgroundColor: '#333',
    borderRadius: '3px',
  },
  hint: {
    fontSize: '0.75rem',
    color: '#666',
    textAlign: 'center',
    padding: '0.5rem',
    backgroundColor: '#f9f9f9',
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.5rem',
  },
  hintIcon: {
    fontSize: '1rem',
  },
};

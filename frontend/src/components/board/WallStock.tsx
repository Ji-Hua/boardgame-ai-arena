import React from 'react';

import { PLAYER_COLORS } from '../../theme/playerColors';

interface WallStockProps {
  player: 'P1' | 'P2';
  wallsRemaining: number;
  orientation: 'horizontal' | 'vertical';
  onEnterWallMode?: () => void;
  isActive?: boolean;
}

/**
 * Wall stock component showing remaining walls for a player
 * Click a wall to enter wall placement mode
 */
export const WallStock: React.FC<WallStockProps> = ({
  player,
  wallsRemaining,
  orientation,
  onEnterWallMode,
  isActive = false,
}) => {
  const color = PLAYER_COLORS[player].primary;
  // Create vertical bar icons
  const wallIcons = Array.from({ length: wallsRemaining }, (_, index) => (
    <div
      key={index}
      style={{
        width: '10px',
        height: '28px',
        backgroundColor: color,
        margin: '0 4px',
        borderRadius: '4px',
        transition: 'all 0.2s ease',
      }}
    />
  ));

  return (
    <div
      onClick={wallsRemaining > 0 ? onEnterWallMode : undefined}
      style={{
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '12px 16px',
        backgroundColor: isActive ? '#fffbea' : '#f5f5f5',
        borderRadius: '8px',
        border: isActive ? `3px solid ${color}` : `2px solid ${color}`,
        cursor: wallsRemaining > 0 ? 'pointer' : 'not-allowed',
        transition: 'all 0.2s ease',
        boxShadow: isActive ? `0 0 12px ${color}40` : 'none',
        opacity: wallsRemaining > 0 ? 1 : 0.5,
        gap: '12px',
      }}
      title={wallsRemaining > 0 ? `Click to place ${player} wall` : 'No walls remaining'}
    >
      <div style={{ fontSize: '14px', fontWeight: 'bold', color }}>
        {player}
      </div>
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'center' }}>
        {wallIcons}
      </div>
      <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#666' }}>
        {wallsRemaining}
      </div>
    </div>
  );
};
"""DQN observation encoder v2 — board-flip normalization for Quoridor.

Extends dqn_obs_v1 with a y-axis flip for P2 so that the network always
observes the game from a consistent orientation: the current player's pawn
moves toward increasing y (goal_y=8 in normalized space).

Version: dqn_obs_v2

P1 (goal_y=8): no transformation — coordinates used as-is.
P2 (goal_y=0): y-flip applied:
  Pawn:      (x, y) → (x, 8-y)      (BOARD_SIZE-1 = 8)
  Wall head: (wx, wy) → (wx, 7-wy)  (WALL_GRID_SIZE-1 = 7)
             Both H-walls and V-walls use the same transform.
             Horizontal stays horizontal, vertical stays vertical
             (a y-flip does not swap wall categories).

The resulting 292-element layout is identical to v1:

  Offset   Length  Description
  ------   ------  -----------
       0       81  Current-player pawn one-hot  (normalized coords)
      81       81  Opponent pawn one-hot         (normalized coords)
     162       64  Horizontal wall-head occupancy (normalized coords)
     226       64  Vertical wall-head occupancy   (normalized coords)
     290        1  Current-player remaining walls (normalized by MAX_WALLS)
     291        1  Opponent remaining walls       (normalized by MAX_WALLS)
  ------   ------
     292      total

Normalization effect:
  P2's starting pawn at (4, 8) appears at normalized slot 36 (4*9+0),
  matching P1's starting position encoding. This lets the network learn
  a single goal-approach policy regardless of which seat the learner
  occupies.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version and size constants (mirroring v1 structure)
# ---------------------------------------------------------------------------

OBSERVATION_VERSION: str = "dqn_obs_v2"

_BOARD_SQUARES: int = 81
_WALL_HEADS: int = 64
_WALL_GRID_SIZE: int = 8
_BOARD_SIZE: int = 9
_MAX_WALLS: int = 10

OBS_OFFSET_CURRENT_PAWN: int = 0
OBS_OFFSET_OPPONENT_PAWN: int = 81
OBS_OFFSET_H_WALLS: int = 162
OBS_OFFSET_V_WALLS: int = 226
OBS_OFFSET_CURRENT_WALLS: int = 290
OBS_OFFSET_OPPONENT_WALLS: int = 291

OBSERVATION_SIZE: int = 292


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode_observation_v2(state: object) -> list[float]:
    """Encode a RawState into a 292-element float list with board-flip for P2.

    For P1, output is identical to ``encode_observation`` (v1).
    For P2, pawn and wall coordinates are y-flipped so that the current
    player always appears to move toward increasing y.

    Parameters
    ----------
    state:
        A ``quoridor_engine.RawState`` instance.

    Returns
    -------
    list[float] of length OBSERVATION_SIZE (292).

    The state is not mutated.
    """
    import quoridor_engine as qe

    obs: list[float] = [0.0] * OBSERVATION_SIZE

    current_player = state.current_player
    opponent = current_player.opponent()

    flip = (current_player == qe.Player.P2)

    # ------------------------------------------------------------------
    # Coordinate transform helpers
    # ------------------------------------------------------------------
    def pawn_idx(x: int, y: int) -> int:
        ny = (_BOARD_SIZE - 1 - y) if flip else y
        return x * _BOARD_SIZE + ny

    def wall_idx(bit_idx: int) -> int:
        if not flip:
            return bit_idx
        wx = bit_idx // _WALL_GRID_SIZE
        wy = bit_idx % _WALL_GRID_SIZE
        nwy = _WALL_GRID_SIZE - 1 - wy
        return wx * _WALL_GRID_SIZE + nwy

    # ------------------------------------------------------------------
    # 1. Current-player pawn one-hot (offset 0, length 81)
    # ------------------------------------------------------------------
    cx, cy = state.pawn_pos(current_player)
    obs[OBS_OFFSET_CURRENT_PAWN + pawn_idx(cx, cy)] = 1.0

    # ------------------------------------------------------------------
    # 2. Opponent pawn one-hot (offset 81, length 81)
    # ------------------------------------------------------------------
    ox, oy = state.pawn_pos(opponent)
    obs[OBS_OFFSET_OPPONENT_PAWN + pawn_idx(ox, oy)] = 1.0

    # ------------------------------------------------------------------
    # 3. Horizontal wall-head occupancy (offset 162, length 64)
    # ------------------------------------------------------------------
    h_heads: int = state.horizontal_heads
    for bit_idx in range(_WALL_HEADS):
        if (h_heads >> bit_idx) & 1:
            obs[OBS_OFFSET_H_WALLS + wall_idx(bit_idx)] = 1.0

    # ------------------------------------------------------------------
    # 4. Vertical wall-head occupancy (offset 226, length 64)
    # ------------------------------------------------------------------
    v_heads: int = state.vertical_heads
    for bit_idx in range(_WALL_HEADS):
        if (v_heads >> bit_idx) & 1:
            obs[OBS_OFFSET_V_WALLS + wall_idx(bit_idx)] = 1.0

    # ------------------------------------------------------------------
    # 5. Remaining walls — normalized by MAX_WALLS (offset 290-291)
    # ------------------------------------------------------------------
    obs[OBS_OFFSET_CURRENT_WALLS] = state.walls_remaining(current_player) / _MAX_WALLS
    obs[OBS_OFFSET_OPPONENT_WALLS] = state.walls_remaining(opponent) / _MAX_WALLS

    return obs


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def observation_size() -> int:
    """Return the canonical observation vector length."""
    return OBSERVATION_SIZE

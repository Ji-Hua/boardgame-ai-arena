"""DQN observation encoder for standard 9x9 Quoridor.

Encodes a RawState into a fixed-length float32 list suitable for use
as a neural-network input. Returns a plain Python list[float] —
trivially convertible to a numpy array or torch.Tensor by the caller.

Version: dqn_obs_v1

Vector layout (292 values total):

  Offset   Length  Description
  ------   ------  -----------
       0       81  Current-player pawn one-hot  (9×9 board, idx = x*9+y)
      81       81  Opponent pawn one-hot         (9×9 board, idx = x*9+y)
     162       64  Horizontal wall-head occupancy (8×8, idx = x*8+y)
     226       64  Vertical wall-head occupancy   (8×8, idx = x*8+y)
     290        1  Current-player remaining walls (normalized by MAX_WALLS)
     291        1  Opponent remaining walls       (normalized by MAX_WALLS)
  ------   ------
     292      total

Current-player-centric encoding:
  "Current player" always refers to state.current_player — the player
  whose turn it is. Pawn planes, wall counts, and the player indicator
  are all expressed relative to that player, not hard-wired to P1/P2.
  This allows the DQN to learn a single policy for either seat.

Board orientation:
  Raw engine coordinates (x, y) are used as-is (origin bottom-left,
  x increases right, y increases upward). No board flip is applied in
  this version. A future encoder version may add orientation mirroring;
  this is tracked in implementation notes.

Wall occupancy:
  Read from state.horizontal_heads and state.vertical_heads bitsets.
  Bit index = x * WALL_GRID_SIZE + y, matching the action_space mapping.

Stability note:
  OBSERVATION_VERSION and OBSERVATION_SIZE are the canonical identifiers
  for this encoding. Checkpoint metadata MUST store both values and
  reject mismatches on load.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version and size constants
# ---------------------------------------------------------------------------

OBSERVATION_VERSION: str = "dqn_obs_v1"

# Layout sizes.
_BOARD_SQUARES: int = 81        # 9 * 9
_WALL_HEADS: int = 64           # 8 * 8
_WALL_GRID_SIZE: int = 8
_BOARD_SIZE: int = 9
_MAX_WALLS: int = 10            # standard initial walls per player

# Segment offsets (kept as named constants for test visibility).
OBS_OFFSET_CURRENT_PAWN: int = 0
OBS_OFFSET_OPPONENT_PAWN: int = 81
OBS_OFFSET_H_WALLS: int = 162
OBS_OFFSET_V_WALLS: int = 226
OBS_OFFSET_CURRENT_WALLS: int = 290
OBS_OFFSET_OPPONENT_WALLS: int = 291

# Total vector length.
OBSERVATION_SIZE: int = 292     # 81 + 81 + 64 + 64 + 1 + 1


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode_observation(state: object) -> list[float]:
    """Encode a RawState into a flat float32-compatible list.

    Parameters
    ----------
    state:
        A ``quoridor_engine.RawState`` instance.

    Returns
    -------
    list[float] of length OBSERVATION_SIZE (292).

    The state is not mutated.
    """
    obs: list[float] = [0.0] * OBSERVATION_SIZE

    current_player = state.current_player
    import quoridor_engine as qe
    opponent = current_player.opponent()

    # ------------------------------------------------------------------
    # 1. Current-player pawn one-hot (offset 0, length 81)
    # ------------------------------------------------------------------
    cx, cy = state.pawn_pos(current_player)
    obs[OBS_OFFSET_CURRENT_PAWN + cx * _BOARD_SIZE + cy] = 1.0

    # ------------------------------------------------------------------
    # 2. Opponent pawn one-hot (offset 81, length 81)
    # ------------------------------------------------------------------
    ox, oy = state.pawn_pos(opponent)
    obs[OBS_OFFSET_OPPONENT_PAWN + ox * _BOARD_SIZE + oy] = 1.0

    # ------------------------------------------------------------------
    # 3. Horizontal wall-head occupancy (offset 162, length 64)
    # ------------------------------------------------------------------
    h_heads: int = state.horizontal_heads
    for bit_idx in range(_WALL_HEADS):
        if (h_heads >> bit_idx) & 1:
            obs[OBS_OFFSET_H_WALLS + bit_idx] = 1.0

    # ------------------------------------------------------------------
    # 4. Vertical wall-head occupancy (offset 226, length 64)
    # ------------------------------------------------------------------
    v_heads: int = state.vertical_heads
    for bit_idx in range(_WALL_HEADS):
        if (v_heads >> bit_idx) & 1:
            obs[OBS_OFFSET_V_WALLS + bit_idx] = 1.0

    # ------------------------------------------------------------------
    # 5. Remaining walls — normalized by MAX_WALLS (offset 290-291)
    # ------------------------------------------------------------------
    obs[OBS_OFFSET_CURRENT_WALLS] = state.walls_remaining(current_player) / _MAX_WALLS
    obs[OBS_OFFSET_OPPONENT_WALLS] = state.walls_remaining(opponent) / _MAX_WALLS

    return obs


# ---------------------------------------------------------------------------
# Convenience helpers (used by tests and env)
# ---------------------------------------------------------------------------

def observation_size() -> int:
    """Return the canonical observation vector length."""
    return OBSERVATION_SIZE

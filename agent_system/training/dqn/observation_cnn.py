"""CNN observation encoder for standard 9x9 Quoridor.

Returns a [C, 9, 9] float32 nested list suitable for use as a
convolutional neural-network input.  Compatible with ``torch.tensor()``
for single-obs inference and list-of-obs batching in the replay buffer.

Version: dqn_obs_cnn_v1

Tensor shape: [7, 9, 9]   (CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)

Channel map (0-indexed):

  Channel  Description
  -------  -----------
        0  Current-player pawn one-hot (9×9)
        1  Opponent pawn one-hot       (9×9)
        2  Horizontal wall-head occupancy (8×8 heads, zero-padded to 9×9)
        3  Vertical wall-head occupancy   (8×8 heads, zero-padded to 9×9)
        4  Current-player remaining walls, broadcast over 9×9
           (normalized by MAX_WALLS = 10)
        5  Opponent remaining walls, broadcast over 9×9
           (normalized by MAX_WALLS = 10)
        6  Current-player goal-row indicator (9×9)
           — entire column at the current player's target row set to 1.0

Current-player-centric semantics:
    "Current player" is always ``state.current_player`` — the player
    whose turn it is.  Channels 0, 1, 4, 5, and 6 are all expressed
    relative to that player so the CNN learns a single policy for
    either seat.

Board coordinates:
    Origin bottom-left; x increases right, y increases upward.
    No board flip applied (same as dqn_obs_v1).

Wall encoding note:
    Wall heads occupy an 8×8 grid (indices 0–7 in each axis).  They
    are placed at the same (x, y) position in the 9×9 plane; row and
    column 8 are always zero (padding).  The bit index formula is
    x * WALL_GRID_SIZE + y, matching action_space.py.

Goal rows:
    P1 wins by reaching y=8; P2 wins by reaching y=0.
    Channel 6 sets all cells in the current player's goal row to 1.0.

Why dqn_obs_v1 is preserved:
    This encoder is independent of observation.py.  The MLP flat encoder
    (dqn_obs_v1) is unchanged and remains the default for MLP training.

Public API:
    encode_observation_cnn(state) -> list[list[list[float]]]   shape [7,9,9]
    CNN_OBSERVATION_VERSION: str
    CNN_CHANNELS: int
    CNN_BOARD_SIZE: int
    CNN_OBSERVATION_SHAPE: tuple[int, int, int]
    CNN_OBSERVATION_SIZE: int
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version and shape constants
# ---------------------------------------------------------------------------

CNN_OBSERVATION_VERSION: str = "dqn_obs_cnn_v1"

CNN_CHANNELS: int = 7
CNN_BOARD_SIZE: int = 9
CNN_OBSERVATION_SHAPE: tuple[int, int, int] = (CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE)
CNN_OBSERVATION_SIZE: int = CNN_CHANNELS * CNN_BOARD_SIZE * CNN_BOARD_SIZE  # 567

_WALL_GRID_SIZE: int = 8
_MAX_WALLS: int = 10


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def encode_observation_cnn(state: object) -> list:
    """Encode a RawState into a [7, 9, 9] nested list of float32 values.

    Parameters
    ----------
    state:
        A ``quoridor_engine.RawState`` instance.

    Returns
    -------
    Nested list of shape [CNN_CHANNELS, CNN_BOARD_SIZE, CNN_BOARD_SIZE]
    (i.e. ``list[list[list[float]]]`` with outer length 7, inner 9×9).
    Compatible with ``torch.tensor(..., dtype=torch.float32)``.

    The state is not mutated.
    """
    # Initialise 7 planes of 9×9 zeros
    planes: list[list[list[float]]] = [
        [[0.0] * CNN_BOARD_SIZE for _ in range(CNN_BOARD_SIZE)]
        for _ in range(CNN_CHANNELS)
    ]

    import quoridor_engine as qe

    current_player = state.current_player
    opponent = current_player.opponent()

    # ------------------------------------------------------------------
    # Channel 0: current-player pawn one-hot
    # ------------------------------------------------------------------
    cx, cy = state.pawn_pos(current_player)
    planes[0][cx][cy] = 1.0

    # ------------------------------------------------------------------
    # Channel 1: opponent pawn one-hot
    # ------------------------------------------------------------------
    ox, oy = state.pawn_pos(opponent)
    planes[1][ox][oy] = 1.0

    # ------------------------------------------------------------------
    # Channel 2: horizontal wall-head occupancy (8×8 → 9×9 zero-padded)
    # ------------------------------------------------------------------
    h_heads: int = state.horizontal_heads
    for bit_idx in range(_WALL_GRID_SIZE * _WALL_GRID_SIZE):
        if (h_heads >> bit_idx) & 1:
            wx = bit_idx // _WALL_GRID_SIZE
            wy = bit_idx % _WALL_GRID_SIZE
            planes[2][wx][wy] = 1.0

    # ------------------------------------------------------------------
    # Channel 3: vertical wall-head occupancy (8×8 → 9×9 zero-padded)
    # ------------------------------------------------------------------
    v_heads: int = state.vertical_heads
    for bit_idx in range(_WALL_GRID_SIZE * _WALL_GRID_SIZE):
        if (v_heads >> bit_idx) & 1:
            wx = bit_idx // _WALL_GRID_SIZE
            wy = bit_idx % _WALL_GRID_SIZE
            planes[3][wx][wy] = 1.0

    # ------------------------------------------------------------------
    # Channel 4: current-player remaining walls, broadcast
    # ------------------------------------------------------------------
    cur_walls_norm: float = state.walls_remaining(current_player) / _MAX_WALLS
    for x in range(CNN_BOARD_SIZE):
        for y in range(CNN_BOARD_SIZE):
            planes[4][x][y] = cur_walls_norm

    # ------------------------------------------------------------------
    # Channel 5: opponent remaining walls, broadcast
    # ------------------------------------------------------------------
    opp_walls_norm: float = state.walls_remaining(opponent) / _MAX_WALLS
    for x in range(CNN_BOARD_SIZE):
        for y in range(CNN_BOARD_SIZE):
            planes[5][x][y] = opp_walls_norm

    # ------------------------------------------------------------------
    # Channel 6: current-player goal-row indicator
    # P1 wins at y=8; P2 wins at y=0.
    # ------------------------------------------------------------------
    goal_y: int = 8 if current_player == qe.Player.P1 else 0
    for x in range(CNN_BOARD_SIZE):
        planes[6][x][goal_y] = 1.0

    return planes


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def observation_shape() -> tuple[int, int, int]:
    """Return the canonical CNN observation tensor shape (C, H, W)."""
    return CNN_OBSERVATION_SHAPE


def observation_size() -> int:
    """Return the total number of float values in the CNN observation."""
    return CNN_OBSERVATION_SIZE

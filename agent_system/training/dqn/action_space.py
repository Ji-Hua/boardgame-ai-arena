"""Fixed DQN action space for standard 9x9 Quoridor.

Action ID layout (209 total, version: dqn_action_v1):
  - 0..80   (81 ids): MovePawn to square (x, y), id = x * 9 + y
  - 81..144 (64 ids): Place horizontal wall at head (x, y), id = 81 + x * 8 + y
  - 145..208 (64 ids): Place vertical wall at head (x, y), id = 145 + x * 8 + y

Coordinate convention:
  Board squares:     x in [0, 8], y in [0, 8]  (9x9 grid)
  Wall head coords:  x in [0, 7], y in [0, 7]  (8x8 grid)
  Origin is bottom-left; x increases right, y increases upward.

This mapping is intentionally stable. Do not change IDs without bumping
ACTION_SPACE_VERSION and updating all downstream consumers.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version and size constants
# ---------------------------------------------------------------------------

ACTION_SPACE_VERSION: str = "dqn_action_v1"

# Total number of discrete action IDs.
ACTION_COUNT: int = 209

# ID range boundaries (inclusive start, exclusive end).
PAWN_ID_START: int = 0
PAWN_ID_END: int = 81       # 9*9 = 81 squares

HWALL_ID_START: int = 81
HWALL_ID_END: int = 145     # 81 + 8*8 = 145

VWALL_ID_START: int = 145
VWALL_ID_END: int = 209     # 145 + 8*8 = 209

# Board dimensions (standard Quoridor).
BOARD_SIZE: int = 9         # squares per side
WALL_GRID_SIZE: int = 8     # wall head positions per side (BOARD_SIZE - 1)


# ---------------------------------------------------------------------------
# Encoding helpers: engine action -> action ID
# ---------------------------------------------------------------------------

def encode_move_pawn(x: int, y: int) -> int:
    """Encode a MovePawn target square (x, y) to an action ID.

    x in [0, BOARD_SIZE-1], y in [0, BOARD_SIZE-1].
    """
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        raise ValueError(
            f"MovePawn target ({x}, {y}) out of bounds for board size {BOARD_SIZE}"
        )
    return PAWN_ID_START + x * BOARD_SIZE + y


def encode_place_hwall(x: int, y: int) -> int:
    """Encode a horizontal wall head (x, y) to an action ID.

    x in [0, WALL_GRID_SIZE-1], y in [0, WALL_GRID_SIZE-1].
    """
    if not (0 <= x < WALL_GRID_SIZE and 0 <= y < WALL_GRID_SIZE):
        raise ValueError(
            f"Horizontal wall head ({x}, {y}) out of bounds for wall grid size {WALL_GRID_SIZE}"
        )
    return HWALL_ID_START + x * WALL_GRID_SIZE + y


def encode_place_vwall(x: int, y: int) -> int:
    """Encode a vertical wall head (x, y) to an action ID.

    x in [0, WALL_GRID_SIZE-1], y in [0, WALL_GRID_SIZE-1].
    """
    if not (0 <= x < WALL_GRID_SIZE and 0 <= y < WALL_GRID_SIZE):
        raise ValueError(
            f"Vertical wall head ({x}, {y}) out of bounds for wall grid size {WALL_GRID_SIZE}"
        )
    return VWALL_ID_START + x * WALL_GRID_SIZE + y


def encode_engine_action(action: object) -> int:
    """Encode an engine Action object to an action ID.

    Reads action.kind, action.target_x, action.target_y, action.coordinate_kind
    as exposed by the PyO3 Action bindings.
    """
    kind = str(action.kind)
    x = int(action.target_x)
    y = int(action.target_y)

    if kind == "MovePawn":
        return encode_move_pawn(x, y)
    elif kind == "PlaceWall":
        coord_kind = str(action.coordinate_kind)
        if coord_kind == "Horizontal":
            return encode_place_hwall(x, y)
        elif coord_kind == "Vertical":
            return encode_place_vwall(x, y)
        else:
            raise ValueError(f"Unknown coordinate_kind: {coord_kind!r}")
    else:
        raise ValueError(f"Unknown action kind: {kind!r}")


# ---------------------------------------------------------------------------
# Decoding helpers: action ID -> engine action components
# ---------------------------------------------------------------------------

def decode_action_id(action_id: int, player: object) -> object:
    """Decode an action ID to an engine Action object.

    player must be a quoridor_engine.Player instance (P1 or P2).
    Raises ValueError for out-of-range action_id.
    """
    _validate_action_id(action_id)

    # Import lazily so this module can be imported before the Rust bindings are built.
    import quoridor_engine as qe

    if PAWN_ID_START <= action_id < PAWN_ID_END:
        offset = action_id - PAWN_ID_START
        x = offset // BOARD_SIZE
        y = offset % BOARD_SIZE
        return qe.Action.move_pawn(player, x, y)

    if HWALL_ID_START <= action_id < HWALL_ID_END:
        offset = action_id - HWALL_ID_START
        x = offset // WALL_GRID_SIZE
        y = offset % WALL_GRID_SIZE
        return qe.Action.place_wall(player, x, y, qe.Orientation.Horizontal)

    # VWALL range (validated above)
    offset = action_id - VWALL_ID_START
    x = offset // WALL_GRID_SIZE
    y = offset % WALL_GRID_SIZE
    return qe.Action.place_wall(player, x, y, qe.Orientation.Vertical)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def is_valid_action_id(action_id: int) -> bool:
    """Return True if action_id is within the valid DQN action space."""
    return isinstance(action_id, int) and 0 <= action_id < ACTION_COUNT


def _validate_action_id(action_id: int) -> None:
    """Raise ValueError if action_id is out of range."""
    if not is_valid_action_id(action_id):
        raise ValueError(
            f"action_id {action_id!r} is out of range [0, {ACTION_COUNT})"
        )


# ---------------------------------------------------------------------------
# Legal action mask and IDs
# ---------------------------------------------------------------------------

def legal_action_mask(engine: object, state: object) -> list[bool]:
    """Build a 209-element boolean mask from engine-authoritative legal actions.

    True at position i means action_id i is legal in the current state.

    engine: quoridor_engine.RuleEngine instance.
    state:  quoridor_engine.RawState instance.
    """
    mask = [False] * ACTION_COUNT
    for action in engine.legal_actions(state):
        action_id = encode_engine_action(action)
        mask[action_id] = True
    return mask


def legal_action_ids(engine: object, state: object) -> list[int]:
    """Return a list of legal action IDs for the current state.

    engine: quoridor_engine.RuleEngine instance.
    state:  quoridor_engine.RawState instance.
    """
    ids = []
    for action in engine.legal_actions(state):
        ids.append(encode_engine_action(action))
    return ids

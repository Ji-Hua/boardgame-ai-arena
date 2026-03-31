"""
Minimal End-to-End Wall Blockage Acceptance Test
=================================================

Translated from:
    quoridor_v0/quoridor-rust-engine/tests-wheel/test_wall_blockage_acceptance.py

This file locks the BEHAVIOR of the legacy acceptance tests in the new
coordinate system defined by:
    documents/engine/design/cooridination.md  (Version 4)
    documents/engine/implementation/rule-engine-implementation.md

This file does NOT implement engine logic. It does NOT require valid imports.
APIs are placeholders. The priority is correct behavior preservation and
coordinate translation.

------------------------------------------------------------------------
Coordinate System — New (Logical, Cartesian)
------------------------------------------------------------------------

    Origin:  (0, 0) at bottom-left
    x:       increases rightward   (column)
    y:       increases upward      (row)
    N = 9

    Squares:  (x, y, square)         0 ≤ x < 9, 0 ≤ y < 9
    H-edges:  (x, y, horizontal)     edge above (x, y), blocks (x,y)↔(x,y+1)
    V-edges:  (x, y, vertical)       edge right of (x,y), blocks (x,y)↔(x+1,y)

    Horizontal wall at head (x, y, h):
        occupies edges (x, y, h) and (x+1, y, h)
        blocks (x, y)↔(x, y+1) AND (x+1, y)↔(x+1, y+1)

    Vertical wall at head (x, y, v):
        occupies edges (x, y, v) and (x, y+1, v)
        blocks (x, y)↔(x+1, y) AND (x, y+1)↔(x+1, y+1)

------------------------------------------------------------------------
Legacy → New Conversion (applied to every coordinate in this file)
------------------------------------------------------------------------

    Square:  P(row, col)  →  (x=col,  y=8-row,  square)
    H-wall:  H(r, c)      →  (x=c,    y=7-r,    horizontal)
    V-wall:  V(r, c)      →  (x=c,    y=7-r,    vertical)

    Player 1 start: legacy (0,4) → (4, 8)   goal: y = 0
    Player 2 start: legacy (8,4) → (4, 0)   goal: y = 8

------------------------------------------------------------------------
Glendenning Display Notation Reference
------------------------------------------------------------------------

    Square (x, y) → file rank   where file = chr(x + ord('a')), rank = y + 1
    Wall   (x, y, ori) → file rank ori   (ori ∈ {h, v})

    Examples used below:
        (4, 8) = e9      (4, 0) = e1
        (2, 7, v) = c8v  (4, 7, v) = e8v
        (3, 6, h) = d7h  (0, 0, h) = a1h
"""

# ---------------------------------------------------------------------------
#  Placeholder imports — not real; the file does not need to run
# ---------------------------------------------------------------------------
# from quoridor_engine import RuleEngine, RawState, Action, Player, Coordinate

N = 9
INITIAL_WALLS_PER_PLAYER = 10

# Player starting positions (new logical coordinates)
P1_START = (4, 8)  # e9 — top-centre,  goal row: y = 0
P2_START = (4, 0)  # e1 — bottom-centre, goal row: y = 8


# ===================================================================
# BLOCK 1 — P1 trapped between two vertical walls
# ===================================================================
#
# Setup (custom state):
#   P1 at (4, 8)  [e9]        P2 at (4, 0)  [e1]
#   Vertical walls: (2, 7, v) [c8v]  and  (4, 7, v) [e8v]
#   P1 walls remaining: 8     P2 walls remaining: 10
#   Current player: 1
#
# Geometry:
#   V(2,7) blocks x=2↔3 at y=7,8  (left barrier)
#   V(4,7) blocks x=4↔5 at y=7,8  (right barrier)
#   P1 at (4,8) is confined to x∈{3,4} for y∈{7,8}
#   Only escape: downward through y=6
#
# Test 1.1  Place horizontal wall (3, 6, h)  [d7h]
#   → ILLEGAL  reason: WALL_BLOCKS_ALL_PATHS
#     Blocks (3,6)↔(3,7) and (4,6)↔(4,7) — seals the downward escape
#   → Turn stays with Player 1
#
# Test 1.2  Place horizontal wall (4, 6, h)  [e7h]
#   → LEGAL
#     Blocks (4,6)↔(4,7) and (5,6)↔(5,7) — leaves x=3 escape open
#   → Turn switches to Player 2
#
# Test 1.3  Place vertical wall (3, 7, v)  [d8v]
#   → LEGAL
#     Blocks x=3↔4 at y=7,8 — P1 can still escape via (4,7)→(4,6)
#   → Turn switches to Player 2
#
# Test 1.4  Place horizontal wall (2, 6, h)  [c7h]
#   → LEGAL
#     Blocks (2,6)↔(2,7) and (3,6)↔(3,7) — leaves x=4 escape open
#   → Turn switches to Player 2


# ===================================================================
# BLOCK 2 — P1 one row down, same vertical wall trap
# ===================================================================
#
# Setup (custom state):
#   P1 at (4, 7)  [e8]        P2 at (4, 0)  [e1]
#   Vertical walls: (2, 7, v) [c8v]  and  (4, 7, v) [e8v]
#   P1 walls remaining: 8     P2 walls remaining: 10
#   Current player: 1
#
# Geometry:
#   Same vertical barriers as Block 1.
#   P1 now at y=7 (one row closer to goal).
#   Confined to x∈{3,4} at y=7,8.  Escape: downward through y=6.
#
# Test 2.1  Place horizontal wall (3, 6, h)  [d7h]
#   → ILLEGAL  reason: WALL_BLOCKS_ALL_PATHS
#     Seals y=6↔7 at x=3,4 — traps P1 in {(3,7),(4,7),(3,8),(4,8)}
#   → Turn stays with Player 1
#
# Test 2.2  Place horizontal wall (3, 7, h)  [d8h]
#   → LEGAL
#     Blocks y=7↔8 at x=3,4 — P1 can still go down from y=7 to y=6
#   → Turn switches to Player 2
#
# Test 2.3  Place horizontal wall (3, 5, h)  [d6h]
#   → LEGAL
#     Blocks y=5↔6 at x=3,4 — below the trap; P1 escapes y=7→y=6 first
#   → Turn switches to Player 2


# ===================================================================
# BLOCK 3 — P1 trapped between horizontal walls (mirror of Block 1)
# ===================================================================
#
# Setup (custom state):
#   P1 at (0, 4)  [a5]        P2 at (8, 4)  [i5]
#   Horizontal walls: (0, 5, h) [a6h]  and  (0, 3, h) [a4h]
#   P1 walls remaining: 8     P2 walls remaining: 10
#   Current player: 1
#
# Geometry:
#   H(0,5) blocks y=5↔6 at x=0,1  (upper barrier)
#   H(0,3) blocks y=3↔4 at x=0,1  (lower barrier)
#   P1 at (0,4) is confined to y∈{4,5} for x∈{0,1}
#   Only escape: rightward through x=2
#
# NOTE: Non-standard player positions.
#   P1 goal: y = 0.   P2 goal: y = 8.
#
# Test 3.1  Place vertical wall (1, 4, v)  [b5v]
#   → ILLEGAL  reason: WALL_BLOCKS_ALL_PATHS
#     Blocks x=1↔2 at y=4,5 — seals the rightward escape
#   → Turn stays with Player 1
#
# Test 3.2  Place vertical wall (1, 3, v)  [b4v]
#   → LEGAL
#     Blocks x=1↔2 at y=3,4 — P1 can still escape via (1,5)→(2,5)
#   → Turn switches to Player 2
#
# Test 3.3  Place vertical wall (1, 5, v)  [b6v]
#   → LEGAL
#     Blocks x=1↔2 at y=5,6 — P1 can still escape via (1,4)→(2,4)
#   → Turn switches to Player 2


# ===================================================================
# BLOCK 4 — Sequential gameplay with illegal move handling
# ===================================================================
#
# Setup: Fresh default game
#   P1 at (4, 8) [e9]   P2 at (4, 0) [e1]
#   No walls placed.  Current player: 1.
#
# Step 4.1  P1 moves pawn to (4, 7) [e8]
#   → LEGAL.  Turn → P2
#
# Step 4.2  P2 places vertical wall (3, 7, v) [d8v]
#   → LEGAL.  Turn → P1
#   Blocks x=3↔4 at y=7,8
#
# Step 4.3  P1 moves pawn to (4, 6) [e7]
#   → LEGAL.  Turn → P2
#
# Step 4.4  P2 places vertical wall (4, 7, v) [e8v]
#   → LEGAL.  Turn → P1
#   Blocks x=4↔5 at y=7,8
#
#   Board state: V(3,7) and V(4,7) form a corridor at x=4 for y=7,8
#
# Step 4.5  P1 moves pawn to (4, 7) [e8]  (moves back up)
#   → LEGAL.  Turn → P2
#
# Step 4.6.1  P2 tries horizontal wall (3, 6, h) [d7h]
#   → ILLEGAL  reason: WALL_BLOCKS_ALL_PATHS
#     Blocks y=6↔7 at x=3,4 — seals P1's only downward escape
#   → Turn stays with Player 2
#
# Step 4.6.2  P2 places horizontal wall (2, 6, h) [c7h]
#   → LEGAL
#     Blocks y=6↔7 at x=2,3 — does not seal P1 (x=4 still open)
#   → Turn → P1


# ===================================================================
# BLOCK 5 — Sequential wall overlap detection
# ===================================================================
#
# Setup: Fresh default game
#   P1 at (4, 8) [e9]   P2 at (4, 0) [e1]
#   No walls placed.  Current player: 1.
#
# All walls in this block are horizontal.
# Overlap occurs when two horizontal walls share an occupied edge
# (a wall at head x occupies edge x and x+1; overlap if adjacent wall
#  head is at x-1 or x+1 so their edge sets intersect).
#
# Step 5.1   P1 places (0, 0, h) [a1h]     → LEGAL.  Turn → P2
#   Occupies edges (0,0,h) and (1,0,h)
#
# Step 5.2   P2 places (2, 0, h) [c1h]     → LEGAL.  Turn → P1
#   Occupies edges (2,0,h) and (3,0,h)
#
# Step 5.3   P1 places (4, 0, h) [e1h]     → LEGAL.  Turn → P2
#   Occupies edges (4,0,h) and (5,0,h)
#
# Step 5.4   P2 places (6, 0, h) [g1h]     → LEGAL.  Turn → P1
#   Occupies edges (6,0,h) and (7,0,h)
#
# Step 5.5   P1 tries (7, 0, h) [h1h]      → ILLEGAL  reason: WALL_OVERLAP
#   Would occupy edges (7,0,h) and (8,0,h)
#   Edge (7,0,h) already occupied by wall from step 5.4
#   → Turn stays with Player 1
#
# Step 5.6   P1 places (7, 1, h) [h2h]     → LEGAL.  Turn → P2
#   Occupies edges (7,1,h) and (8,1,h)
#
# Step 5.7   P2 tries (6, 1, h) [g2h]      → ILLEGAL  reason: WALL_OVERLAP
#   Would occupy edges (6,1,h) and (7,1,h)
#   Edge (7,1,h) already occupied by wall from step 5.6
#   → Turn stays with Player 2
#
# Step 5.8   P2 places (5, 1, h) [f2h]     → LEGAL.  Turn → P1
#   Occupies edges (5,1,h) and (6,1,h)
#
# Step 5.9   P1 tries (4, 1, h) [e2h]      → ILLEGAL  reason: WALL_OVERLAP
#   Would occupy edges (4,1,h) and (5,1,h)
#   Edge (5,1,h) already occupied by wall from step 5.8
#   → Turn stays with Player 1
#
# Step 5.10  P1 places (3, 1, h) [d2h]     → LEGAL.  Turn → P2
#   Occupies edges (3,1,h) and (4,1,h)


# ===================================================================
# Structured test outline (placeholder — not runnable)
# ===================================================================

def test_wall_blockage_minimal_e2e():
    """
    Single minimal end-to-end behaviour case covering:
      - Wall-blocked path detection  (WALL_BLOCKS_ALL_PATHS)
      - Wall overlap detection        (WALL_OVERLAP)
      - Turn preservation on illegal actions
      - Turn switch on legal actions
      - Sequential multi-turn gameplay

    All coordinates use the new logical system (origin bottom-left).
    """

    # --- helpers (pseudo) ---
    # engine = RuleEngine(...)
    # def make_state(p1, p2, h_walls, v_walls, current, p1_rem, p2_rem) -> RawState
    # def place_wall(state, player, x, y, orientation) -> (RawState | RuleError)
    # def move_pawn(state, player, x, y) -> (RawState | RuleError)

    # ---------------------------------------------------------------
    # BLOCK 1 — vertical wall trap
    # ---------------------------------------------------------------
    state = "make_state(p1=(4,8), p2=(4,0), h_walls=[], v_walls=[(2,7),(4,7)], current=1, p1_rem=8, p2_rem=10)"

    # 1.1 — (3, 6, h) d7h → ILLEGAL WALL_BLOCKS_ALL_PATHS, turn stays P1
    assert "place_wall(state, 1, 3, 6, horizontal)" == "RuleError(WALL_BLOCKS_ALL_PATHS)"
    # state unchanged, current_player == 1

    # 1.2 — (4, 6, h) e7h → LEGAL, turn → P2
    assert "place_wall(state, 1, 4, 6, horizontal)" == "Ok(new_state)"
    # current_player == 2

    # 1.3 — (reset) (3, 7, v) d8v → LEGAL, turn → P2
    assert "place_wall(state, 1, 3, 7, vertical)" == "Ok(new_state)"

    # 1.4 — (reset) (2, 6, h) c7h → LEGAL, turn → P2
    assert "place_wall(state, 1, 2, 6, horizontal)" == "Ok(new_state)"

    # ---------------------------------------------------------------
    # BLOCK 2 — P1 one row closer to goal
    # ---------------------------------------------------------------
    state = "make_state(p1=(4,7), p2=(4,0), h_walls=[], v_walls=[(2,7),(4,7)], current=1, p1_rem=8, p2_rem=10)"

    # 2.1 — (3, 6, h) d7h → ILLEGAL WALL_BLOCKS_ALL_PATHS, turn stays P1
    assert "place_wall(state, 1, 3, 6, horizontal)" == "RuleError(WALL_BLOCKS_ALL_PATHS)"

    # 2.2 — (reset) (3, 7, h) d8h → LEGAL
    assert "place_wall(state, 1, 3, 7, horizontal)" == "Ok(new_state)"

    # 2.3 — (reset) (3, 5, h) d6h → LEGAL
    assert "place_wall(state, 1, 3, 5, horizontal)" == "Ok(new_state)"

    # ---------------------------------------------------------------
    # BLOCK 3 — horizontal wall trap (mirror)
    # ---------------------------------------------------------------
    state = "make_state(p1=(0,4), p2=(8,4), h_walls=[(0,5),(0,3)], v_walls=[], current=1, p1_rem=8, p2_rem=10)"

    # 3.1 — (1, 4, v) b5v → ILLEGAL WALL_BLOCKS_ALL_PATHS, turn stays P1
    assert "place_wall(state, 1, 1, 4, vertical)" == "RuleError(WALL_BLOCKS_ALL_PATHS)"

    # 3.2 — (reset) (1, 3, v) b4v → LEGAL
    assert "place_wall(state, 1, 1, 3, vertical)" == "Ok(new_state)"

    # 3.3 — (reset) (1, 5, v) b6v → LEGAL
    assert "place_wall(state, 1, 1, 5, vertical)" == "Ok(new_state)"

    # ---------------------------------------------------------------
    # BLOCK 4 — sequential gameplay
    # ---------------------------------------------------------------
    state = "fresh_game()"
    # P1 at (4,8), P2 at (4,0), current=1

    # 4.1  P1 pawn → (4, 7) e8       LEGAL, turn → P2
    state = "move_pawn(state, 1, 4, 7)"

    # 4.2  P2 wall V(3, 7) d8v       LEGAL, turn → P1
    state = "place_wall(state, 2, 3, 7, vertical)"

    # 4.3  P1 pawn → (4, 6) e7       LEGAL, turn → P2
    state = "move_pawn(state, 1, 4, 6)"

    # 4.4  P2 wall V(4, 7) e8v       LEGAL, turn → P1
    state = "place_wall(state, 2, 4, 7, vertical)"

    # 4.5  P1 pawn → (4, 7) e8       LEGAL (back up), turn → P2
    state = "move_pawn(state, 1, 4, 7)"

    # 4.6.1  P2 wall H(3, 6) d7h     ILLEGAL WALL_BLOCKS_ALL_PATHS, turn stays P2
    assert "place_wall(state, 2, 3, 6, horizontal)" == "RuleError(WALL_BLOCKS_ALL_PATHS)"

    # 4.6.2  P2 wall H(2, 6) c7h     LEGAL (retry), turn → P1
    state = "place_wall(state, 2, 2, 6, horizontal)"

    # ---------------------------------------------------------------
    # BLOCK 5 — sequential wall overlap
    # ---------------------------------------------------------------
    state = "fresh_game()"

    # 5.1   P1 H(0, 0) a1h           LEGAL, turn → P2
    state = "place_wall(state, 1, 0, 0, horizontal)"

    # 5.2   P2 H(2, 0) c1h           LEGAL, turn → P1
    state = "place_wall(state, 2, 2, 0, horizontal)"

    # 5.3   P1 H(4, 0) e1h           LEGAL, turn → P2
    state = "place_wall(state, 1, 4, 0, horizontal)"

    # 5.4   P2 H(6, 0) g1h           LEGAL, turn → P1
    state = "place_wall(state, 2, 6, 0, horizontal)"

    # 5.5   P1 H(7, 0) h1h           ILLEGAL WALL_OVERLAP, turn stays P1
    assert "place_wall(state, 1, 7, 0, horizontal)" == "RuleError(WALL_OVERLAP)"

    # 5.6   P1 H(7, 1) h2h           LEGAL, turn → P2
    state = "place_wall(state, 1, 7, 1, horizontal)"

    # 5.7   P2 H(6, 1) g2h           ILLEGAL WALL_OVERLAP, turn stays P2
    assert "place_wall(state, 2, 6, 1, horizontal)" == "RuleError(WALL_OVERLAP)"

    # 5.8   P2 H(5, 1) f2h           LEGAL, turn → P1
    state = "place_wall(state, 2, 5, 1, horizontal)"

    # 5.9   P1 H(4, 1) e2h           ILLEGAL WALL_OVERLAP, turn stays P1
    assert "place_wall(state, 1, 4, 1, horizontal)" == "RuleError(WALL_OVERLAP)"

    # 5.10  P1 H(3, 1) d2h           LEGAL, turn → P2
    state = "place_wall(state, 1, 3, 1, horizontal)"

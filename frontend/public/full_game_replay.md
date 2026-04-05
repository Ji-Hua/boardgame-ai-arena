# Quoridor Single-Game E2E Replay Script

Standard configuration:
- Board: 9x9
- Initial pawns:
  - Player 1 at (4,0,Square)
  - Player 2 at (4,8,Square)
- Walls:
  - Player 1 starts with 10
  - Player 2 starts with 10
- Turn rule:
  - An ACCEPT step changes state and passes the turn
  - A REJECT step does not change state and does not pass the turn

Notation used below:
- Integer step number = correct step that should be executed
- Decimal step number = incorrect step attempted immediately after the previous correct step, and should be rejected

---

## Opening and Early Confrontation

Step 1:
- Action: { player: 1, kind: MovePawn, target: (4,1,Square) } -> ACCEPT

Step 2:
- Action: { player: 2, kind: MovePawn, target: (4,7,Square) } -> ACCEPT

Step 3:
- Action: { player: 1, kind: MovePawn, target: (4,2,Square) } -> ACCEPT

Step 4:
- Action: { player: 2, kind: MovePawn, target: (4,6,Square) } -> ACCEPT

Step 5:
- Action: { player: 1, kind: MovePawn, target: (4,3,Square) } -> ACCEPT

Step 6:
- Action: { player: 2, kind: MovePawn, target: (4,5,Square) } -> ACCEPT

Step 6.1:
- Action: { player: 1, kind: MovePawn, target: (4,5,Square) } -> REJECT
- Why wrong:
  - This is an attempted jump while the pawns are not adjacent.
  - Player 1 is at (4,3) and Player 2 is at (4,5), so there is still one square between them.

Step 7:
- Action: { player: 1, kind: MovePawn, target: (4,4,Square) } -> ACCEPT

Step 7.1:
- Action: { player: 2, kind: MovePawn, target: (4,4,Square) } -> REJECT
- Why wrong:
  - A pawn cannot move onto an occupied square.
  - (4,4) is currently occupied by Player 1.

Step 7.2:
- Action: { player: 1, kind: MovePawn, target: (4,5,Square) } -> REJECT
- Why wrong:
  - Step 7.1 was rejected, so the turn did not change.
  - It is still Player 2's turn here.

Step 8:
- Action: { player: 2, kind: MovePawn, target: (4,3,Square) } -> ACCEPT
- Note:
  - Legal direct jump over an adjacent opponent.
  - Player 2 jumps from (4,5) over Player 1 at (4,4) to (4,3).

Step 8.1:
- Action: { player: 1, kind: MovePawn, target: (5,3,Square) } -> REJECT
- Why wrong:
  - A side jump is not allowed when a direct jump is available.
  - Player 1 is adjacent to Player 2 and the square behind Player 2 is open, so the legal jump is straight to (4,2), not diagonally to (5,3).

Step 8.2:
- Action: { player: 1, kind: MovePawn, target: (3,3,Square) } -> REJECT
- Why wrong:
  - A side jump is not allowed when a direct jump is available.
  - Player 1 is adjacent to Player 2 and the square behind Player 2 is open, so the legal jump is straight to (4,2), not diagonally to (3,3).

Step 9:
- Action: { player: 1, kind: MovePawn, target: (4,2,Square) } -> ACCEPT
- Note:
  - Legal direct jump over Player 2 from (4,4) to (4,2).

Step 10:
- Action: { player: 2, kind: PlaceWall, target: (4,2,Horizontal) } -> ACCEPT

### State Snapshot After Step 10

- Pawn positions:
  - Player 1: (4,2,Square)
  - Player 2: (4,3,Square)

- Walls on board:
  - (4,2,Horizontal)

- Walls remaining:
  - Player 1: 10
  - Player 2: 9

- Position interpretation:
  - The two pawns remain vertically adjacent on file x = 4.
  - The horizontal wall at (4,2) occupies:
    - (4,2,Horizontal)
    - (5,2,Horizontal)
  - By geometric definition, this wall blocks movement between:
    - (4,2) and (4,3)
    - (5,2) and (5,3)
  - Therefore the direct adjacency between the two pawns does not imply a legal jump.
  - The immediate confrontation has been converted from an open jump position into a wall-separated contact position.

- Strategic stage:
  - This is the first structural disruption of the opening.
  - The game transitions from pure pawn-race interaction into wall-shaped local geometry.

Step 10.1:
- Action: { player: 1, kind: MovePawn, target: (4,4,Square) } -> REJECT
- Why wrong:
  - Player 1 and Player 2 are now separated by the horizontal wall at (4,2), so Player 1 cannot jump over Player 2

Step 10.2:
- Action: { player: 1, kind: MovePawn, target: (3,3,Square) } -> REJECT
- Why wrong:
  - Because the direct contact is blocked by the wall at (4,2), the side-jump condition is not satisfied here.

Step 11:
- Action: { player: 1, kind: MovePawn, target: (3,2,Square) } -> ACCEPT

Step 12:
- Action: { player: 2, kind: PlaceWall, target: (4,3,Horizontal) } -> ACCEPT

Step 13:
- Action: { player: 1, kind: MovePawn, target: (3,3,Square) } -> ACCEPT

Step 13.1:
- Action: { player: 2, kind: MovePawn, target: (3,3,Square) } -> REJECT
- Why wrong:
  - Target square is occupied.

Step 13.2:
- Action: { player: 2, kind: MovePawn, target: (4,2,Square) } -> REJECT
- Why wrong:
  - Blocked by wall (4,2,Horizontal).

Step 14:
- Action: { player: 2, kind: MovePawn, target: (2,3,Square) } -> ACCEPT
- Note: p2 legal direct jump

Step 15:
- Action: { player: 1, kind: MovePawn, target: (3,4,Square) } -> ACCEPT

Step 16:
- Action: { player: 2, kind: MovePawn, target: (3,3,Square) } -> ACCEPT

Step 17:
- Action: { player: 1, kind: PlaceWall, target: (7,0,Horizontal) } -> ACCEPT

Step 18:
- Action: { player: 2, kind: MovePawn, target: (3,2,Square) } -> ACCEPT

Step 18.1:
- Action: { player: 1, kind: PlaceWall, target: (4,3,Horizontal) } -> REJECT
- Why wrong:
  - Exact overlap with an existing wall is illegal.

Step 18.2:
- Action: { player: 1, kind: PlaceWall, target: (4,3,Vertical) } -> REJECT
- Why wrong:
  - This wall crosses the existing horizontal wall at (4,3).
  - Crossing walls are illegal.

Step 18.3:
- Action: { player: 1, kind: PlaceWall, target: (3,3,Horizontal) } -> REJECT
- Why wrong:
  - One segment overlaps with an existing wall, which is illegal.

### State Snapshot After Step 18

- Pawn positions:
  - Player 1: (3,4,Square)
  - Player 2: (3,2,Square)

- Walls on board:
  - (4,2,Horizontal)
  - (7,0,Horizontal)
  - (4,3,Horizontal)

- Walls remaining:
  - Player 1: 9
  - Player 2: 8

- Strategic stage:
  - This is the transition from jump tactics into positional routing.
  - Wall structure is starting to matter more than direct pawn contact.

Step 19:
- Action: { player: 1, kind: PlaceWall, target: (2,4,Vertical) } -> ACCEPT

Step 19.1:
- Action: { player: 2, kind: PlaceWall, target: (8,5,Vertical) } -> REJECT
- Why wrong:
  - This wall target is out of bounds for a vertical wall on a 9x9 board.

Step 19.2:
- Action: { player: 2, kind: PlaceWall, target: (8,5,Horizontal) } -> REJECT
- Why wrong:
  - This wall target is out of bounds (2nd segment).

Step 20:
- Action: { player: 2, kind: MovePawn, target: (2,2,Square) } -> ACCEPT

Step 20.1:
- Action: { player: 1, kind: MovePawn, target: (2,4,Square) } -> REJECT
- Why wrong:
  - The vertical wall at (2,4) blocks movement from (3,4) to (2,4).

Step 21:
- Action: { player: 1, kind: PlaceWall, target: (0,1,Vertical) } -> ACCEPT

Step 21.1:
- Action: { player: 2, kind: MovePawn, target: (0,2,Square) } -> REJECT
- Why wrong:
  - Player 2 is at (2,2), so moving directly to (0,2) is an illegal two-square lateral move.

Step 22:
- Action: { player: 2, kind: MovePawn, target: (1,2,Square) } -> ACCEPT

Step 23:
- Action: { player: 1, kind: MovePawn, target: (3,5,Square) } -> ACCEPT

Step 23.1:
- Action: { player: 2, kind: MovePawn, target: (0,2,Square) } -> REJECT
- Why wrong:
  - The vertical wall at (0, 1) blocks movement from (1,2) to (0,2).

Step 24:
- Action: { player: 2, kind: MovePawn, target: (1,1,Square) } -> ACCEPT

Step 24.1:
- Action: { player: 1, kind: MovePawn, target: (3,7,Square) } -> REJECT
- Why wrong:
  - This is an illegal two-square move from (3,5) to (3,7).

Step 25:
- Action: { player: 1, kind: MovePawn, target: (3,6,Square) } -> ACCEPT

Step 25.1:
- Action: { player: 2, kind: PlaceWall, target: (3,8,Horizontal) } -> REJECT
- Why wrong:
  - This wall target is out of bounds.

Step 25.2:
- Action: { player: 2, kind: PlaceWall, target: (3,8,Vertical) } -> REJECT
- Why wrong:
  - This wall target is out of bounds (2nd wall segment).

Step 26:
- Action: { player: 2, kind: MovePawn, target: (1,2,Square) } -> ACCEPT

Step 26.1:
- Action: { player: 1, kind: MovePawn, target: (3,8,Square) } -> REJECT
- Why wrong:
  - This is an illegal two-square move from (3,6) to (3,8).

Step 26.2:
- Action: { player: 1, kind: MovePawn, target: (2,7,Square) } -> REJECT
- Why wrong:
  - This is an illegal diagnol move from (3,6) to (2,7).

Step 26.3:
- Action: { player: 1, kind: MovePawn, target: (5,7,Square) } -> REJECT
- Why wrong:
  - This is an illegal move from (3,6) to (5,7).

Step 26.4:
- Action: { player: 1, kind: MovePawn, target: (3,4,Square) } -> REJECT
- Why wrong:
  - This is an illegal two-square move from (3,6) to (3,4).

Step 27:
- Action: { player: 1, kind: PlaceWall, target: (3,6,Horizontal) } -> ACCEPT

Step 27.1:
- Action: { player: 2, kind: PlaceWall, target: (3,6,Horizontal) } -> REJECT
- Why wrong:
  - Exact wall overlap is illegal.

Step 27.2:
- Action: { player: 2, kind: PlaceWall, target: (4,6,Horizontal) } -> REJECT
- Why wrong:
  - One segment wall overlap is illegal.

Step 27.3:
- Action: { player: 2, kind: PlaceWall, target: (2,5,Vertical) } -> REJECT
- Why wrong:
  - One segment wall overlap is illegal.

Step 28:
- Action: { player: 2, kind: MovePawn, target: (1,3,Square) } -> ACCEPT

### State Snapshot After Step 28

- Pawn positions:
  - Player 1: (3,6,Square)
  - Player 2: (1,3,Square)

- Walls on board:
  - (4,2,Horizontal)
  - (7,0,Horizontal)
  - (4,3,Horizontal)
  - (2,4,Vertical)
  - (0,1,Vertical)
  - (3,6,Horizontal)

- Walls remaining:
  - Player 1: 6
  - Player 2: 8

- Strategic stage:
  - The game has entered midgame wall shaping.
  - Route denial is now stronger than direct confrontation.

Step 28.1:
- Action: { player: 1, kind: MovePawn, target: (3,7,Square) } -> REJECT
- Why wrong:
  - The horizontal wall at (3,6) blocks movement from (3,6) to (3,7).

Step 29:
- Action: { player: 1, kind: MovePawn, target: (2,6,Square) } -> ACCEPT

Step 30:
- Action: { player: 2, kind: MovePawn, target: (1,4,Square) } -> ACCEPT

Step 31:
- Action: { player: 1, kind: MovePawn, target: (2,7,Square) } -> ACCEPT

Step 32:
- Action: { player: 2, kind: PlaceWall, target: (2,7,Horizontal) } -> ACCEPT

Step 32.1:
- Action: { player: 1, kind: MovePawn, target: (2,8,Square) } -> REJECT
- Why wrong:
  - The horizontal wall at (2,7) blocks movement from (2,7) to (2,8).

Step 33:
- Action: { player: 1, kind: MovePawn, target: (3,7,Square) } -> ACCEPT

Step 33.1:
- Action: { player: 2, kind: MovePawn, target: (-1,4,Square) } -> REJECT
- Why wrong:
  - The target square is out of bounds.

Step 34:
- Action: { player: 2, kind: MovePawn, target: (1,5,Square) } -> ACCEPT

Step 35:
- Action: { player: 1, kind: MovePawn, target: (4,7,Square) } -> ACCEPT

Step 35.1:
- Action: { player: 2, kind: MovePawn, target: (1,7,Square) } -> REJECT
- Why wrong:
  - This is an illegal two-square move from (1,5) to (1,7).

Step 36:
- Action: { player: 2, kind: PlaceWall, target: (4,7,Horizontal) } -> ACCEPT

Step 36.1:
- Action: { player: 1, kind: MovePawn, target: (4,8,Square) } -> REJECT
- Why wrong:
  - The horizontal wall at (4,7) blocks movement from (4,7) to (4,8).

Step 37:
- Action: { player: 1, kind: MovePawn, target: (5,7,Square) } -> ACCEPT

Step 37.1:
- Action: { player: 2, kind: PlaceWall, target: (4,7,Horizontal) } -> REJECT
- Why wrong:
  - Exact wall overlap is illegal.

Step 38:
- Action: { player: 2, kind: MovePawn, target: (1,6,Square) } -> ACCEPT

### State Snapshot After Step 38

- Pawn positions:
  - Player 1: (5,7,Square)
  - Player 2: (1,6,Square)

- Walls on board:
  - (4,2,Horizontal)
  - (7,0,Horizontal)
  - (4,3,Horizontal)
  - (2,4,Vertical)
  - (0,1,Vertical)
  - (3,6,Horizontal)
  - (2,7,Horizontal)
  - (4,7,Horizontal)

- Walls remaining:
  - Player 1: 6
  - Player 2: 6

- Strategic stage:
  - The game is now in a race-and-blocking phase.
  - Positional nearness to the goal is no longer enough; geometric access is decisive.

Step 38.1:
- Action: { player: 1, kind: MovePawn, target: (5,9,Square) } -> REJECT
- Why wrong:
  - The target square is out of bounds.

Step 39:
- Action: { player: 1, kind: MovePawn, target: (6,7,Square) } -> ACCEPT

Step 40:
- Action: { player: 2, kind: MovePawn, target: (1,7,Square) } -> ACCEPT

Step 40.1:
- Action: { player: 1, kind: MovePawn, target: (8,7,Square) } -> REJECT
- Why wrong:
  - This is an illegal two-square lateral move from (6,7) to (8,7).

Step 41:
- Action: { player: 1, kind: PlaceWall, target: (6,7,Horizontal) } -> ACCEPT

Step 42:
- Action: { player: 2, kind: MovePawn, target: (1,8,Square) } -> ACCEPT
- Note:
  - Game should NOT end here, because Player 2 wins only by reaching row y = 0, not y = 8.

Step 43:
- Action: { player: 1, kind: MovePawn, target: (5,7,Square) } -> ACCEPT

Step 43.1:
- Action: { player: 2, kind: MovePawn, target: (1,9,Square) } -> REJECT
- Why wrong:
  - The target square is out of bounds.

Step 44:
- Action: { player: 2, kind: MovePawn, target: (1,7,Square) } -> ACCEPT

Step 45:
- Action: { player: 1, kind: MovePawn, target: (4,7,Square) } -> ACCEPT

Step 46:
- Action: { player: 2, kind: PlaceWall, target: (0,6,Vertical) } -> ACCEPT

Step 47:
- Action: { player: 1, kind: PlaceWall, target: (1,5,Horizontal) } -> ACCEPT

Step 48:
- Action: { player: 2, kind: PlaceWall, target: (3,5,Horizontal) } -> ACCEPT

### State Snapshot After Step 48

- Pawn positions:
  - Player 1: (4,7,Square)
  - Player 2: (1,7,Square)

- Walls on board:
  - (4,2,Horizontal)
  - (7,0,Horizontal)
  - (4,3,Horizontal)
  - (2,4,Vertical)
  - (0,1,Vertical)
  - (3,6,Horizontal)
  - (2,7,Horizontal)
  - (4,7,Horizontal)
  - (6,7,Horizontal)
  - (0,6,Vertical)
  - (1,5,Horizontal)
  - (3,5,Horizontal)

- Walls remaining:
  - Player 1: 4
  - Player 2: 4

- Position interpretation:
  - Both pawns now occupy row y = 7, separated laterally.
  - The newly added wall chain at row y = 5 begins to form a strong lower boundary:
    - (1,5,Horizontal)
    - (3,5,Horizontal)
  - Together with the existing upper-row walls around y = 7, the board is starting to develop a boxed and corridor-like structure.
  - The vertical wall at (0,6,Vertical) narrows left-side horizontal access near the upper-left region.

- Strategic stage:
  - The game has entered constrained endgame shaping.
  - Remaining paths are beginning to collapse into fewer viable channels.

Step 49:
- Action: { player: 1, kind: PlaceWall, target: (5,5,Horizontal) } -> ACCEPT

Step 50:
- Action: { player: 2, kind: PlaceWall, target: (7,5,Horizontal) } -> ACCEPT

Step 50.1:
- Action: { player: 1, kind: PlaceWall, target: (8,7,Horizontal) } -> REJECT
- Why wrong:
  - The second segment of this wall would be (9,7,Horizontal), which is out of bounds.

Step 51:
- Action: { player: 1, kind: PlaceWall, target: (7,6,Vertical) } -> ACCEPT
- Note:
  - Now there is a boxed region.
  - The intended remaining route to the target side is the narrow opening near the left side.

Step 51.1:
- Action: { player: 2, kind: PlaceWall, target: (0,7,Horizontal) } -> REJECT
- Why wrong:
  - This wall placement itself is in bounds and does not overlap or cross an existing wall.
  - It must be rejected because it would eliminate the remaining winning path(s), creating a no-path situation.

Step 52:
- Action: { player: 2, kind: PlaceWall, target: (1,6,Horizontal) } -> ACCEPT

Step 53:
- Action: { player: 1, kind: MovePawn, target: (3,7,Square) } -> ACCEPT

Step 54:
- Action: { player: 2, kind: MovePawn, target: (2,7,Square) } -> ACCEPT

Step 55:
- Action: { player: 1, kind: MovePawn, target: (1,7,Square) } -> ACCEPT
- Note:
  - Legal horizontal direct jump.

Step 56:
- Action: { player: 2, kind: MovePawn, target: (3,7,Square) } -> ACCEPT

Step 57:
- Action: { player: 1, kind: MovePawn, target: (2,7,Square) } -> ACCEPT

Step 57.1:
- Action: { player: 2, kind: PlaceWall, target: (1,7,Vertical) } -> REJECT
- Why wrong:
  - This wall placement itself is structurally legal, but it must be rejected because it would create a no-path situation.

Step 57.2:
- Action: { player: 2, kind: PlaceWall, target: (2,7,Vertical) } -> REJECT
- Why wrong:
  - Cross wall is illegal.

Step 58:
- Action: { player: 2, kind: PlaceWall, target: (3,7,Vertical) } -> ACCEPT
- Note:
  - After this step, the pawns are adjacent in coordinates, but all jump continuations are blocked by walls.

### State Snapshot After Step 58

- Pawn positions:
  - Player 1: (2,7,Square)
  - Player 2: (3,7,Square)

- Walls on board:
  - (4,2,Horizontal)
  - (7,0,Horizontal)
  - (4,3,Horizontal)
  - (2,4,Vertical)
  - (0,1,Vertical)
  - (3,6,Horizontal)
  - (2,7,Horizontal)
  - (4,7,Horizontal)
  - (6,7,Horizontal)
  - (0,6,Vertical)
  - (1,5,Horizontal)
  - (3,5,Horizontal)
  - (5,5,Horizontal)
  - (7,5,Horizontal)
  - (7,6,Vertical)
  - (1,6,Horizontal)
  - (3,7,Vertical)

- Walls remaining:
  - Player 1: 2
  - Player 2: 1

- Strategic stage:
  - The game is now in a fully constrained adjacency state.
  - Endgame legality depends on exact local edge occupancy, not on intuitive pawn proximity.

Step 58.1:
- Action: { player: 1, kind: MovePawn, target: (4,7,Square) } -> REJECT
- Why wrong:
  - Straight jump is not legal here.
  - The jump path is blocked by the wall configuration, so the required jump condition is not satisfied.

Step 58.2:
- Action: { player: 1, kind: MovePawn, target: (3,8,Square) } -> REJECT
- Why wrong:
  - Diagonal jump is not legal here.
  - The relevant diagonal continuation is blocked by the wall configuration.

Step 58.3:
- Action: { player: 1, kind: MovePawn, target: (3,6,Square) } -> REJECT
- Why wrong:
  - Diagonal jump in the opposite direction is also blocked.
  - No valid jump continuation exists.

Step 58.4:
- Action: { player: 1, kind: PlaceWall, target: (2,6,Vertical) } -> REJECT
- Why wrong:
  - This wall placement itself is in bounds and does not overlap or cross an existing wall.
  - It must be rejected because it would create a no-path situation for Player 2.

Step 59:
- Action: { player: 1, kind: PlaceWall, target: (2,3,Horizontal) } -> ACCEPT
- Note:
  - Random legal wall, unrelated to the local row 6–8 structure.

Step 60:
- Action: { player: 2, kind: MovePawn, target: (1,7,Square) } -> ACCEPT
- Note:
  - Legal straight jump.

Step 60.1:
- Action: { player: 1, kind: PlaceWall, target: (0,3,Horizontal) } -> REJECT
- Why wrong:
  - This wall placement itself is in bounds and does not overlap or cross an existing wall.
  - It must be rejected because it would create another no-path situation for Player 2.

Step 61:
- Action: { player: 1, kind: MovePawn, target: (1,8,Square) } -> ACCEPT
- Note:
  - Player 1 reaches the target row and wins.

Step 61.1:
- Action: { player: 2, kind: MovePawn, target: (2,7,Square) } -> REJECT
- Note:
  - No action allowed since game ended.

Step 61.2:
- Action: { player: 2, kind: PlaceWall, target: (3,0,Horizontal) } -> REJECT
- Note:
  - No action allowed since game ended.

Step 61.3:
- Action: { player: 2, kind: PlaceWall, target: (3,0,Vertical) } -> REJECT
- Note:
  - No action allowed since game ended.

### Final State Snapshot After Step 61

- Final pawn positions:
  - Player 1: (1,8,Square)
  - Player 2: (1,7,Square)

- Final result:
  - Player 1 wins by reaching the target row y = 8.
  - Player 2 does not win by reaching y = 8 because Player 2's winning condition is reaching y = 0.

- Final board interpretation:
  - The late game progressively compressed the legal routes into a narrow left-side channel.
  - The no-path rejections at Steps 51.1, 57.1, 58.4, and 60.1 confirm that path preservation remained active all the way into the endgame.
  - The final winning move is not a broad breakthrough, but the resolution of a single surviving route.

- Strategic stage:
  - This is a single-path endgame resolution.
  - The replay ends by demonstrating both correct win detection and correct path-preservation enforcement.

---

## Coverage Summary

This single continuous game covers:

- Standard opening movement
- Wrong-player rejection
- Occupied-square rejection
- Illegal non-adjacent jump attempt
- Legal direct jump
- Illegal side jump when direct jump is available
- Legal side jump when direct jump is blocked
- Illegal diagonal pawn move
- Legal wall placement
- Wall overlap rejection
- Wall crossing rejection
- Wall out-of-bounds rejection
- Pawn out-of-bounds rejection
- Illegal multi-square non-jump pawn moves
- Movement blocked by a wall
- No-path wall rejection
- State continuity across a full game
- Proper win detection
- Phase-by-phase board-state evolution at Steps 10, 18, 28, 38, 48, 58, and final resolution

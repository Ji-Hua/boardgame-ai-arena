/// Calculation utilities — BFS pathfinding and connectivity.
///
/// Provides algorithmic support for rule validation.
/// These are stateless functions over RawState + Topology.

use std::collections::VecDeque;
use crate::model::RawState;
use crate::topology::Topology;
use crate::model::Player;

/// Directions for movement: (dx, dy).
const DIRECTIONS: [(i32, i32); 4] = [
    (1, 0),   // right (+x)
    (-1, 0),  // left  (-x)
    (0, 1),   // up    (+y)
    (0, -1),  // down  (-y)
];

/// Check whether movement from (x, y) in direction (dx, dy) is blocked by a wall.
///
/// Uses dual-edge checks: a wall head occupies two edges, so movement across
/// a single grid boundary can be blocked by either of two wall heads.
///
/// Horizontal edge (ex, ey, h) is above square (ex, ey), blocking (ex, ey) ↔ (ex, ey+1).
/// Vertical edge (ex, ey, v) is right of square (ex, ey), blocking (ex, ey) ↔ (ex+1, ey).
///
/// A horizontal wall at head (wx, wy) sets edges (wx, wy, h) and (wx+1, wy, h).
/// A vertical wall at head (wx, wy) sets edges (wx, wy, v) and (wx, wy+1, v).
///
/// Movement UP from (x, y) to (x, y+1) crosses horizontal edge (x, y, h).
/// That edge is set by wall head (x, y) OR wall head (x-1, y).
///
/// Movement RIGHT from (x, y) to (x+1, y) crosses vertical edge (x, y, v).
/// That edge is set by wall head (x, y) OR wall head (x, y-1).
pub fn is_movement_blocked(state: &RawState, x: usize, y: usize, dx: i32, dy: i32, topo: &Topology) -> bool {
    let n = topo.n();
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;

    // Board boundary check
    if nx < 0 || nx >= n as i32 || ny < 0 || ny >= n as i32 {
        return true;
    }

    if dy == 1 {
        // Moving UP: crosses horizontal edge (x, y, h)
        // This edge is occupied if h_edge(x, y) set — from wall head (x, y)
        // OR h_edge at (x, y) was set as second edge of wall head (x-1, y): h_edge(x, y).
        // Actually, since wall head (hx, hy) sets h_edges at (hx, hy) and (hx+1, hy),
        // h_edge(x, y) is set if ANY wall head with hx=x or hx=x-1 at hy=y was placed.
        // We just check the edge directly since we store edges, not heads.
        if y < n - 1 && state.has_h_edge(x, y, n) {
            return true;
        }
    } else if dy == -1 {
        // Moving DOWN: crosses horizontal edge (x, y-1, h)
        if y > 0 && state.has_h_edge(x, y - 1, n) {
            return true;
        }
    } else if dx == 1 {
        // Moving RIGHT: crosses vertical edge (x, y, v)
        if x < n - 1 && state.has_v_edge(x, y, n) {
            return true;
        }
    } else if dx == -1 {
        // Moving LEFT: crosses vertical edge (x-1, y, v)
        if x > 0 && state.has_v_edge(x - 1, y, n) {
            return true;
        }
    }

    false
}

/// BFS reachability: does a path exist from the player's position to any goal cell?
pub fn path_exists(state: &RawState, player: Player, topo: &Topology) -> bool {
    let n = topo.n();
    let (sx, sy) = state.pawn_pos(player);
    let goal_y = topo.goal_y(player);

    let mut visited = vec![vec![false; n]; n];
    let mut queue = VecDeque::new();

    visited[sx][sy] = true;
    queue.push_back((sx, sy));

    while let Some((x, y)) = queue.pop_front() {
        if y == goal_y {
            return true;
        }
        for &(dx, dy) in &DIRECTIONS {
            if is_movement_blocked(state, x, y, dx, dy, topo) {
                continue;
            }
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;
            if !visited[nx][ny] {
                visited[nx][ny] = true;
                queue.push_back((nx, ny));
            }
        }
    }
    false
}

/// BFS shortest path length from player's position to goal row.
pub fn shortest_path_len(state: &RawState, player: Player, topo: &Topology) -> Option<u32> {
    let n = topo.n();
    let (sx, sy) = state.pawn_pos(player);
    let goal_y = topo.goal_y(player);

    let mut dist: Vec<Vec<Option<u32>>> = vec![vec![None; n]; n];
    let mut queue = VecDeque::new();

    dist[sx][sy] = Some(0);
    queue.push_back((sx, sy));

    while let Some((x, y)) = queue.pop_front() {
        let d = dist[x][y].unwrap();
        if y == goal_y {
            return Some(d);
        }
        for &(dx, dy) in &DIRECTIONS {
            if is_movement_blocked(state, x, y, dx, dy, topo) {
                continue;
            }
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;
            if dist[nx][ny].is_none() {
                dist[nx][ny] = Some(d + 1);
                queue.push_back((nx, ny));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Player, RawState};
    use crate::topology::Topology;

    fn standard_topo() -> Topology {
        Topology::standard()
    }

    fn initial_state(topo: &Topology) -> RawState {
        let (p1x, p1y) = topo.p1_start();
        let (p2x, p2y) = topo.p2_start();
        RawState {
            pawn_positions: [(p1x, p1y), (p2x, p2y)],
            horizontal_edges: 0,
            vertical_edges: 0,
            horizontal_heads: 0,
            vertical_heads: 0,
            remaining_walls: [10, 10],
            current_player: Player::P1,
        }
    }

    fn state_with_walls(
        p1: (usize, usize),
        p2: (usize, usize),
        h_wall_heads: &[(usize, usize)],
        v_wall_heads: &[(usize, usize)],
    ) -> RawState {
        let n = 9;
        let mut state = RawState {
            pawn_positions: [p1, p2],
            horizontal_edges: 0,
            vertical_edges: 0,
            horizontal_heads: 0,
            vertical_heads: 0,
            remaining_walls: [10 - h_wall_heads.len() as u8 - v_wall_heads.len() as u8, 10],
            current_player: Player::P1,
        };
        for &(x, y) in h_wall_heads {
            state.set_h_edge(x, y, n);
            state.set_h_edge(x + 1, y, n);
            state.set_h_head(x, y, n);
        }
        for &(x, y) in v_wall_heads {
            state.set_v_edge(x, y, n);
            state.set_v_edge(x, y + 1, n);
            state.set_v_head(x, y, n);
        }
        state
    }

    #[test]
    fn test_no_walls_path_exists() {
        let topo = standard_topo();
        let state = initial_state(&topo);
        assert!(path_exists(&state, Player::P1, &topo));
        assert!(path_exists(&state, Player::P2, &topo));
    }

    #[test]
    fn test_no_walls_shortest_path() {
        let topo = standard_topo();
        let state = initial_state(&topo);
        // P1 at (4, 0), goal y=8 → 8 steps straight up
        assert_eq!(shortest_path_len(&state, Player::P1, &topo), Some(8));
        // P2 at (4, 8), goal y=0 → 8 steps straight down
        assert_eq!(shortest_path_len(&state, Player::P2, &topo), Some(8));
    }

    #[test]
    fn test_movement_blocked_by_h_wall() {
        let topo = standard_topo();
        // Place horizontal wall at head (3, 4, h) → edges (3, 4, h) and (4, 4, h)
        // Blocks (3, 4)↔(3, 5) and (4, 4)↔(4, 5)
        let state = state_with_walls((4, 8), (4, 0), &[(3, 4)], &[]);

        // Moving UP from (3, 4) to (3, 5) should be blocked
        assert!(is_movement_blocked(&state, 3, 4, 0, 1, &topo));
        // Moving UP from (4, 4) to (4, 5) should be blocked
        assert!(is_movement_blocked(&state, 4, 4, 0, 1, &topo));
        // Moving DOWN from (3, 5) to (3, 4) should be blocked
        assert!(is_movement_blocked(&state, 3, 5, 0, -1, &topo));
        // Moving UP from (2, 4) to (2, 5) should NOT be blocked
        assert!(!is_movement_blocked(&state, 2, 4, 0, 1, &topo));
        // Moving UP from (5, 4) to (5, 5) should NOT be blocked
        assert!(!is_movement_blocked(&state, 5, 4, 0, 1, &topo));
    }

    #[test]
    fn test_movement_blocked_by_v_wall() {
        let topo = standard_topo();
        // Place vertical wall at head (4, 3, v) → edges (4, 3, v) and (4, 4, v)
        // Blocks (4, 3)↔(5, 3) and (4, 4)↔(5, 4)
        let state = state_with_walls((4, 8), (4, 0), &[], &[(4, 3)]);

        // Moving RIGHT from (4, 3) to (5, 3) should be blocked
        assert!(is_movement_blocked(&state, 4, 3, 1, 0, &topo));
        // Moving RIGHT from (4, 4) to (5, 4) should be blocked
        assert!(is_movement_blocked(&state, 4, 4, 1, 0, &topo));
        // Moving LEFT from (5, 3) to (4, 3) should be blocked
        assert!(is_movement_blocked(&state, 5, 3, -1, 0, &topo));
        // Moving RIGHT from (4, 2) to (5, 2) should NOT be blocked
        assert!(!is_movement_blocked(&state, 4, 2, 1, 0, &topo));
    }

    #[test]
    fn test_board_boundary_blocking() {
        let topo = standard_topo();
        let state = initial_state(&topo);
        // Can't move beyond board edges
        assert!(is_movement_blocked(&state, 0, 0, -1, 0, &topo));  // left edge
        assert!(is_movement_blocked(&state, 0, 0, 0, -1, &topo));  // bottom edge
        assert!(is_movement_blocked(&state, 8, 8, 1, 0, &topo));   // right edge
        assert!(is_movement_blocked(&state, 8, 8, 0, 1, &topo));   // top edge
    }

    #[test]
    fn test_vertical_wall_trap_blocks_path() {
        let topo = standard_topo();
        // V(2, 7) and V(4, 7) with H(3, 6) traps P2 at (4, 7)
        // P2 goal is y=0; trap prevents reaching it
        let state = state_with_walls(
            (4, 0), (4, 7),
            &[(3, 6)],         // H wall that seals the trap
            &[(2, 7), (4, 7)], // V walls forming the corridor
        );
        assert!(!path_exists(&state, Player::P2, &topo));
    }

    #[test]
    fn test_vertical_wall_trap_without_seal_has_path() {
        let topo = standard_topo();
        // Same V walls but no H wall — P2 can escape downward
        let state = state_with_walls(
            (4, 0), (4, 7),
            &[],
            &[(2, 7), (4, 7)],
        );
        assert!(path_exists(&state, Player::P2, &topo));
    }
}

/// Rule Engine — stateless rule kernel.
///
/// (RawState, Action) → RawState' | RuleError
///
/// Holds only Topology. No mutable state. Deterministic.

use crate::topology::Topology;
use crate::model::*;
use crate::error::*;
use crate::calculation;

#[derive(Debug, Clone)]
pub struct RuleEngine {
    pub topology: Topology,
}

impl RuleEngine {
    pub fn new(topology: Topology) -> Self {
        Self { topology }
    }

    pub fn standard() -> Self {
        Self::new(Topology::standard())
    }

    // -----------------------------------------------------------------
    // State Bootstrap
    // -----------------------------------------------------------------

    pub fn initial_state(&self) -> RawState {
        let (p1x, p1y) = self.topology.p1_start();
        let (p2x, p2y) = self.topology.p2_start();
        let w = self.topology.initial_walls_per_player;
        RawState {
            pawn_positions: [(p1x, p1y), (p2x, p2y)],
            horizontal_edges: 0,
            vertical_edges: 0,
            horizontal_heads: 0,
            vertical_heads: 0,
            remaining_walls: [w, w],
            current_player: Player::P1,
        }
    }

    // -----------------------------------------------------------------
    // Transition API
    // -----------------------------------------------------------------

    /// Validate and apply an action, producing a new RawState.
    pub fn apply_action(&self, raw: &RawState, action: &Action) -> RuleResult<RawState> {
        self.validate_action(raw, action)?;

        let mut next = raw.clone();
        match action.kind {
            ActionKind::MovePawn => {
                next.pawn_positions[action.player.index()] = (action.target.x, action.target.y);
            }
            ActionKind::PlaceWall => {
                let n = self.topology.n();
                match action.target.kind {
                    CoordinateKind::Horizontal => {
                        next.set_h_edge(action.target.x, action.target.y, n);
                        next.set_h_edge(action.target.x + 1, action.target.y, n);
                        next.set_h_head(action.target.x, action.target.y, n);
                    }
                    CoordinateKind::Vertical => {
                        next.set_v_edge(action.target.x, action.target.y, n);
                        next.set_v_edge(action.target.x, action.target.y + 1, n);
                        next.set_v_head(action.target.x, action.target.y, n);
                    }
                    CoordinateKind::Square => unreachable!(),
                }
                next.remaining_walls[action.player.index()] -= 1;
            }
        }
        next.current_player = raw.current_player.opponent();
        Ok(next)
    }

    /// Validate an action without producing new state.
    pub fn validate_action(&self, raw: &RawState, action: &Action) -> RuleResult<()> {
        // Game-over check
        if self.is_game_over(raw) {
            return Err(RuleError::new(RuleErrorCode::GameOver, "Game is already over"));
        }

        // Turn check
        if action.player != raw.current_player {
            return Err(RuleError::new(RuleErrorCode::WrongTurn, "Not this player's turn"));
        }

        match action.kind {
            ActionKind::MovePawn => self.validate_pawn_move(raw, action),
            ActionKind::PlaceWall => self.validate_wall_placement(raw, action),
        }
    }

    /// Return all legal actions for the current player.
    pub fn legal_actions(&self, raw: &RawState) -> Vec<Action> {
        let player = raw.current_player;
        let mut actions = Vec::new();
        let n = self.topology.n();

        // Candidate pawn moves
        for x in 0..n {
            for y in 0..n {
                let a = Action::move_pawn(player, x, y);
                if self.validate_action(raw, &a).is_ok() {
                    actions.push(a);
                }
            }
        }

        // Candidate wall placements
        for x in 0..n {
            for y in 0..n {
                for &ori in &[Orientation::Horizontal, Orientation::Vertical] {
                    let a = Action::place_wall(player, x, y, ori);
                    if self.validate_action(raw, &a).is_ok() {
                        actions.push(a);
                    }
                }
            }
        }

        actions
    }

    // -----------------------------------------------------------------
    // Query API
    // -----------------------------------------------------------------

    pub fn is_game_over(&self, raw: &RawState) -> bool {
        self.winner(raw).is_some()
    }

    pub fn winner(&self, raw: &RawState) -> Option<Player> {
        let (_, p1y) = raw.pawn_pos(Player::P1);
        let (_, p2y) = raw.pawn_pos(Player::P2);
        if p1y == self.topology.p1_goal_y() {
            Some(Player::P1)
        } else if p2y == self.topology.p2_goal_y() {
            Some(Player::P2)
        } else {
            None
        }
    }

    pub fn path_exists(&self, raw: &RawState, player: Player) -> bool {
        calculation::path_exists(raw, player, &self.topology)
    }

    // -----------------------------------------------------------------
    // Pawn Movement Validation
    // -----------------------------------------------------------------

    fn validate_pawn_move(&self, raw: &RawState, action: &Action) -> RuleResult<()> {
        let (tx, ty) = (action.target.x, action.target.y);

        // Target kind must be Square
        if action.target.kind != CoordinateKind::Square {
            return Err(RuleError::new(RuleErrorCode::InvalidActionKind, "Pawn move target must be Square"));
        }

        // Bounds
        if !self.topology.is_valid_square(tx, ty) {
            return Err(RuleError::new(RuleErrorCode::PawnMoveOutOfBounds, "Target out of bounds"));
        }

        let (fx, fy) = raw.pawn_pos(action.player);
        let (ox, oy) = raw.opponent_pos(action.player);

        // Can't stay in place
        if tx == fx && ty == fy {
            return Err(RuleError::new(RuleErrorCode::PawnMoveSameSquare, "Cannot stay in place"));
        }

        // Can't land on opponent
        if tx == ox && ty == oy {
            return Err(RuleError::new(RuleErrorCode::PawnMoveOccupied, "Target occupied by opponent"));
        }

        let dx = tx as i32 - fx as i32;
        let dy = ty as i32 - fy as i32;
        let adx = dx.unsigned_abs() as usize;
        let ady = dy.unsigned_abs() as usize;

        match (adx, ady) {
            (1, 0) | (0, 1) => self.check_simple_move(raw, fx, fy, dx, dy),
            (2, 0) | (0, 2) => self.check_jump(raw, fx, fy, dx, dy, ox, oy),
            (1, 1) => self.check_sidestep(raw, fx, fy, tx, ty, ox, oy),
            _ => Err(RuleError::new(RuleErrorCode::PawnMoveNotAdjacent, "Move too far")),
        }
    }

    fn check_simple_move(&self, raw: &RawState, fx: usize, fy: usize, dx: i32, dy: i32) -> RuleResult<()> {
        if calculation::is_movement_blocked(raw, fx, fy, dx, dy, &self.topology) {
            return Err(RuleError::new(RuleErrorCode::PawnMoveBlocked, "Wall blocks this move"));
        }
        // Check if opponent is at destination (simple move can't jump over)
        let nx = (fx as i32 + dx) as usize;
        let ny = (fy as i32 + dy) as usize;
        let (ox, oy) = raw.opponent_pos(raw.current_player);
        if nx == ox && ny == oy {
            return Err(RuleError::new(RuleErrorCode::PawnMoveOccupied, "Target occupied by opponent"));
        }
        Ok(())
    }

    fn check_jump(&self, raw: &RawState, fx: usize, fy: usize, dx: i32, dy: i32, ox: usize, oy: usize) -> RuleResult<()> {
        // Direction unit vector
        let udx = dx.signum();
        let udy = dy.signum();

        // Opponent must be at the midpoint
        let mx = (fx as i32 + udx) as usize;
        let my = (fy as i32 + udy) as usize;
        if ox != mx || oy != my {
            return Err(RuleError::new(RuleErrorCode::PawnMoveNotAdjacent, "No opponent to jump over"));
        }

        // Check wall between current → opponent
        if calculation::is_movement_blocked(raw, fx, fy, udx, udy, &self.topology) {
            return Err(RuleError::new(RuleErrorCode::PawnMoveBlocked, "Wall blocks path to opponent"));
        }

        // Check wall between opponent → landing
        if calculation::is_movement_blocked(raw, mx, my, udx, udy, &self.topology) {
            return Err(RuleError::new(RuleErrorCode::PawnMoveBlocked, "Wall blocks jump landing"));
        }

        Ok(())
    }

    fn check_sidestep(&self, raw: &RawState, fx: usize, fy: usize, tx: usize, ty: usize, ox: usize, oy: usize) -> RuleResult<()> {
        // Opponent must be adjacent to current player
        let opp_dx = ox as i32 - fx as i32;
        let opp_dy = oy as i32 - fy as i32;
        if opp_dx.unsigned_abs() as usize + opp_dy.unsigned_abs() as usize != 1 {
            return Err(RuleError::new(RuleErrorCode::PawnMoveNotAdjacent, "Opponent not adjacent"));
        }

        // Target must be adjacent to opponent
        let t_opp_dx = tx as i32 - ox as i32;
        let t_opp_dy = ty as i32 - oy as i32;
        if t_opp_dx.unsigned_abs() as usize + t_opp_dy.unsigned_abs() as usize != 1 {
            return Err(RuleError::new(RuleErrorCode::PawnMoveNotAdjacent, "Target not adjacent to opponent"));
        }

        // Target must be perpendicular to player→opponent direction
        // (the diagonal component)
        if opp_dx != 0 {
            // Opponent is horizontal from player; target must differ in y from opponent
            if t_opp_dx != 0 {
                return Err(RuleError::new(RuleErrorCode::PawnMoveNotAdjacent, "Sidestep must be perpendicular"));
            }
        } else {
            // Opponent is vertical from player; target must differ in x from opponent
            if t_opp_dy != 0 {
                return Err(RuleError::new(RuleErrorCode::PawnMoveNotAdjacent, "Sidestep must be perpendicular"));
            }
        }

        // The direct jump over opponent must be BLOCKED (wall or board edge)
        // Otherwise, the player must jump, not sidestep
        let jump_blocked = calculation::is_movement_blocked(raw, ox, oy, opp_dx, opp_dy, &self.topology);
        if !jump_blocked {
            return Err(RuleError::new(RuleErrorCode::PawnMoveNotAdjacent, "Direct jump available; sidestep not allowed"));
        }

        // Path from player → opponent must not be wall-blocked
        if calculation::is_movement_blocked(raw, fx, fy, opp_dx, opp_dy, &self.topology) {
            return Err(RuleError::new(RuleErrorCode::PawnMoveBlocked, "Wall blocks path to opponent"));
        }

        // Path from opponent → target must not be wall-blocked
        if calculation::is_movement_blocked(raw, ox, oy, t_opp_dx, t_opp_dy, &self.topology) {
            return Err(RuleError::new(RuleErrorCode::PawnMoveBlocked, "Wall blocks sidestep to target"));
        }

        Ok(())
    }

    // -----------------------------------------------------------------
    // Wall Placement Validation
    // -----------------------------------------------------------------

    fn validate_wall_placement(&self, raw: &RawState, action: &Action) -> RuleResult<()> {
        let (x, y) = (action.target.x, action.target.y);
        let n = self.topology.n();

        // Must be Horizontal or Vertical coordinate kind
        let orientation = match action.target.kind {
            CoordinateKind::Horizontal => Orientation::Horizontal,
            CoordinateKind::Vertical => Orientation::Vertical,
            CoordinateKind::Square => {
                return Err(RuleError::new(RuleErrorCode::InvalidActionKind, "Wall target must be Horizontal or Vertical"));
            }
        };

        // Bounds check
        match orientation {
            Orientation::Horizontal => {
                if !self.topology.is_valid_h_wall_head(x, y) {
                    return Err(RuleError::new(RuleErrorCode::WallOutOfBounds, "Horizontal wall out of bounds"));
                }
            }
            Orientation::Vertical => {
                if !self.topology.is_valid_v_wall_head(x, y) {
                    return Err(RuleError::new(RuleErrorCode::WallOutOfBounds, "Vertical wall out of bounds"));
                }
            }
        }

        // Remaining walls check
        if raw.walls_remaining(action.player) == 0 {
            return Err(RuleError::new(RuleErrorCode::NoWallsRemaining, "No walls remaining"));
        }

        // Overlap check: proposed wall's edges must not already be occupied
        match orientation {
            Orientation::Horizontal => {
                // Wall head (x, y, h) occupies edges (x, y, h) and (x+1, y, h)
                if raw.has_h_edge(x, y, n) || raw.has_h_edge(x + 1, y, n) {
                    return Err(RuleError::new(RuleErrorCode::WallOverlap, "WALL_OVERLAP"));
                }
            }
            Orientation::Vertical => {
                // Wall head (x, y, v) occupies edges (x, y, v) and (x, y+1, v)
                if raw.has_v_edge(x, y, n) || raw.has_v_edge(x, y + 1, n) {
                    return Err(RuleError::new(RuleErrorCode::WallOverlap, "WALL_OVERLAP"));
                }
            }
        }

        // Crossing check: a new wall crosses an existing wall if they share
        // the same head position in opposite orientations.
        match orientation {
            Orientation::Horizontal => {
                if raw.has_v_head(x, y, n) {
                    return Err(RuleError::new(RuleErrorCode::WallCrossing, "WALL_CROSSING"));
                }
            }
            Orientation::Vertical => {
                if raw.has_h_head(x, y, n) {
                    return Err(RuleError::new(RuleErrorCode::WallCrossing, "WALL_CROSSING"));
                }
            }
        }

        // Path preservation check: temporarily add wall, verify both players can reach goals
        let mut temp = raw.clone();
        match orientation {
            Orientation::Horizontal => {
                temp.set_h_edge(x, y, n);
                temp.set_h_edge(x + 1, y, n);
            }
            Orientation::Vertical => {
                temp.set_v_edge(x, y, n);
                temp.set_v_edge(x, y + 1, n);
            }
        }

        if !calculation::path_exists(&temp, Player::P1, &self.topology) {
            return Err(RuleError::new(RuleErrorCode::WallBlocksAllPaths, "WALL_BLOCKS_ALL_PATHS"));
        }
        if !calculation::path_exists(&temp, Player::P2, &self.topology) {
            return Err(RuleError::new(RuleErrorCode::WallBlocksAllPaths, "WALL_BLOCKS_ALL_PATHS"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> RuleEngine {
        RuleEngine::standard()
    }

    fn custom_state(
        p1: (usize, usize),
        p2: (usize, usize),
        h_wall_heads: &[(usize, usize)],
        v_wall_heads: &[(usize, usize)],
        current: Player,
        p1_rem: u8,
        p2_rem: u8,
    ) -> RawState {
        let n = 9;
        let mut state = RawState {
            pawn_positions: [p1, p2],
            horizontal_edges: 0,
            vertical_edges: 0,
            horizontal_heads: 0,
            vertical_heads: 0,
            remaining_walls: [p1_rem, p2_rem],
            current_player: current,
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

    // --- Initial state ---

    #[test]
    fn test_initial_state() {
        let e = engine();
        let s = e.initial_state();
        assert_eq!(s.pawn_pos(Player::P1), (4, 0)); // e1
        assert_eq!(s.pawn_pos(Player::P2), (4, 8)); // e9
        assert_eq!(s.current_player, Player::P1);
        assert_eq!(s.walls_remaining(Player::P1), 10);
        assert_eq!(s.walls_remaining(Player::P2), 10);
    }

    // --- Simple pawn moves ---

    #[test]
    fn test_simple_pawn_move_up() {
        let e = engine();
        let s = e.initial_state();
        // P1 at (4, 0) moves to (4, 1) — one step up
        let a = Action::move_pawn(Player::P1, 4, 1);
        let s2 = e.apply_action(&s, &a).unwrap();
        assert_eq!(s2.pawn_pos(Player::P1), (4, 1));
        assert_eq!(s2.current_player, Player::P2);
    }

    #[test]
    fn test_wrong_turn_rejected() {
        let e = engine();
        let s = e.initial_state();
        let a = Action::move_pawn(Player::P2, 4, 1);
        let r = e.validate_action(&s, &a);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, RuleErrorCode::WrongTurn);
    }

    #[test]
    fn test_move_blocked_by_wall() {
        let e = engine();
        // H wall at (3, 6) blocks (3,6)↔(3,7) and (4,6)↔(4,7)
        let s = custom_state((4, 7), (4, 1), &[(3, 6)], &[], Player::P1, 9, 10);
        // P1 at (4, 7) tries to move down to (4, 6) — blocked by h-edge (4, 6)
        let a = Action::move_pawn(Player::P1, 4, 6);
        let r = e.validate_action(&s, &a);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, RuleErrorCode::PawnMoveBlocked);
    }

    // --- Wall placement: overlap ---

    #[test]
    fn test_wall_overlap_same_orientation() {
        let e = engine();
        // Place H wall at (0, 0), then try H wall at (1, 0) which shares edge (1, 0, h)
        let s = custom_state((4, 7), (4, 1), &[(0, 0)], &[], Player::P1, 9, 10);
        // H(1, 0) would need edges (1, 0, h) and (2, 0, h). Edge (1, 0, h) is already occupied.
        let a = Action::place_wall(Player::P1, 1, 0, Orientation::Horizontal);
        let r = e.validate_action(&s, &a);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, RuleErrorCode::WallOverlap);
    }

    #[test]
    fn test_wall_no_overlap_gap() {
        let e = engine();
        // H(0, 0) occupies edges (0,0,h) and (1,0,h). H(2, 0) occupies (2,0,h) and (3,0,h). No overlap.
        let s = custom_state((4, 7), (4, 1), &[(0, 0)], &[], Player::P1, 9, 10);
        let a = Action::place_wall(Player::P1, 2, 0, Orientation::Horizontal);
        assert!(e.validate_action(&s, &a).is_ok());
    }

    // --- Wall placement: crossing ---

    #[test]
    fn test_wall_crossing() {
        let e = engine();
        // H wall at (3, 3), then try V(3, 3) — same intersection point
        let s = custom_state((4, 7), (4, 1), &[(3, 3)], &[], Player::P1, 9, 10);
        let a = Action::place_wall(Player::P1, 3, 3, Orientation::Vertical);
        let r = e.validate_action(&s, &a);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, RuleErrorCode::WallCrossing);
    }

    // --- Wall placement: path blocking ---

    #[test]
    fn test_wall_blocks_all_paths() {
        let e = engine();
        // V walls (2, 7) and (4, 7) trap P2 at (4, 7) in corridor x=3..4, y=7..8
        // P2 goal is y=0; trap prevents reaching it
        let s = custom_state((4, 0), (4, 7), &[], &[(2, 7), (4, 7)], Player::P1, 8, 10);
        // H(3, 6) seals the corridor bottom: blocks y=6↔7 at x=3,4
        let a = Action::place_wall(Player::P1, 3, 6, Orientation::Horizontal);
        let r = e.validate_action(&s, &a);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, RuleErrorCode::WallBlocksAllPaths);
    }

    #[test]
    fn test_wall_does_not_block_all_paths() {
        let e = engine();
        // Same V walls but H(4, 6) only blocks x=4,5 — P2 can escape through x=3
        let s = custom_state((4, 0), (4, 7), &[], &[(2, 7), (4, 7)], Player::P1, 8, 10);
        let a = Action::place_wall(Player::P1, 4, 6, Orientation::Horizontal);
        assert!(e.validate_action(&s, &a).is_ok());
    }

    // --- No walls remaining ---

    #[test]
    fn test_no_walls_remaining() {
        let e = engine();
        let s = custom_state((4, 7), (4, 1), &[], &[], Player::P1, 0, 10);
        let a = Action::place_wall(Player::P1, 0, 0, Orientation::Horizontal);
        let r = e.validate_action(&s, &a);
        assert!(r.is_err());
        assert_eq!(r.unwrap_err().code, RuleErrorCode::NoWallsRemaining);
    }

    // --- Game over / winner ---

    #[test]
    fn test_game_over_p1_wins() {
        let e = engine();
        // P1 at y=8 (goal)
        let s = custom_state((4, 8), (4, 1), &[], &[], Player::P2, 10, 10);
        assert!(e.is_game_over(&s));
        assert_eq!(e.winner(&s), Some(Player::P1));
    }

    #[test]
    fn test_game_over_p2_wins() {
        let e = engine();
        // Both at goal — P1 wins (checked first)
        let s = custom_state((4, 8), (4, 0), &[], &[], Player::P1, 10, 10);
        assert!(e.is_game_over(&s));

        // Only P2 at goal
        let s2 = custom_state((4, 7), (4, 0), &[], &[], Player::P1, 10, 10);
        assert!(e.is_game_over(&s2));
        assert_eq!(e.winner(&s2), Some(Player::P2));
    }

    #[test]
    fn test_not_game_over() {
        let e = engine();
        let s = e.initial_state();
        assert!(!e.is_game_over(&s));
        assert_eq!(e.winner(&s), None);
    }
}

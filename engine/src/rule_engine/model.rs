/// Core model types — pure data, no rule logic.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    P1,
    P2,
}

impl Player {
    pub fn opponent(self) -> Player {
        match self {
            Player::P1 => Player::P2,
            Player::P2 => Player::P1,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Player::P1 => 0,
            Player::P2 => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionKind {
    MovePawn,
    PlaceWall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coordinate {
    pub x: usize,
    pub y: usize,
    pub kind: CoordinateKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateKind {
    Square,
    Horizontal,
    Vertical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Action {
    pub player: Player,
    pub kind: ActionKind,
    pub target: Coordinate,
}

impl Action {
    pub fn move_pawn(player: Player, x: usize, y: usize) -> Self {
        Self {
            player,
            kind: ActionKind::MovePawn,
            target: Coordinate { x, y, kind: CoordinateKind::Square },
        }
    }

    pub fn place_wall(player: Player, x: usize, y: usize, orientation: Orientation) -> Self {
        let kind = match orientation {
            Orientation::Horizontal => CoordinateKind::Horizontal,
            Orientation::Vertical => CoordinateKind::Vertical,
        };
        Self {
            player,
            kind: ActionKind::PlaceWall,
            target: Coordinate { x, y, kind },
        }
    }
}

/// Canonical rule-relevant game state. Immutable by convention.
///
/// Wall storage uses u128 bitsets for occupied EDGES (not wall heads).
/// - Horizontal edges: N × (N-1) positions, index = x * (N-1) + y
/// - Vertical edges: (N-1) × N positions, index = x * N + y
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawState {
    pub pawn_positions: [(usize, usize); 2],
    pub horizontal_edges: u128,
    pub vertical_edges: u128,
    pub horizontal_heads: u128,
    pub vertical_heads: u128,
    pub remaining_walls: [u8; 2],
    pub current_player: Player,
}

impl RawState {
    pub fn pawn_pos(&self, player: Player) -> (usize, usize) {
        self.pawn_positions[player.index()]
    }

    pub fn opponent_pos(&self, player: Player) -> (usize, usize) {
        self.pawn_positions[player.opponent().index()]
    }

    pub fn walls_remaining(&self, player: Player) -> u8 {
        self.remaining_walls[player.index()]
    }
}

/// Edge query and mutation helpers.
/// Board size N is passed explicitly to keep RawState lean.
impl RawState {
    pub fn has_h_edge(&self, x: usize, y: usize, n: usize) -> bool {
        let idx = x * (n - 1) + y;
        (self.horizontal_edges & (1u128 << idx)) != 0
    }

    pub fn has_v_edge(&self, x: usize, y: usize, n: usize) -> bool {
        let idx = x * n + y;
        (self.vertical_edges & (1u128 << idx)) != 0
    }

    pub fn set_h_edge(&mut self, x: usize, y: usize, n: usize) {
        let idx = x * (n - 1) + y;
        self.horizontal_edges |= 1u128 << idx;
    }

    pub fn set_v_edge(&mut self, x: usize, y: usize, n: usize) {
        let idx = x * n + y;
        self.vertical_edges |= 1u128 << idx;
    }
}

/// Wall head query and mutation helpers.
/// Head index: x * (N-1) + y, for x in 0..N-1, y in 0..N-1.
impl RawState {
    pub fn has_h_head(&self, x: usize, y: usize, n: usize) -> bool {
        let idx = x * (n - 1) + y;
        (self.horizontal_heads & (1u128 << idx)) != 0
    }

    pub fn has_v_head(&self, x: usize, y: usize, n: usize) -> bool {
        let idx = x * (n - 1) + y;
        (self.vertical_heads & (1u128 << idx)) != 0
    }

    pub fn set_h_head(&mut self, x: usize, y: usize, n: usize) {
        let idx = x * (n - 1) + y;
        self.horizontal_heads |= 1u128 << idx;
    }

    pub fn set_v_head(&mut self, x: usize, y: usize, n: usize) {
        let idx = x * (n - 1) + y;
        self.vertical_heads |= 1u128 << idx;
    }
}

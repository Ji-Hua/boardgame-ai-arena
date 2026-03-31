/// Board topology — geometric structure independent of rule logic.
///
/// Defines board dimensions, valid positions, adjacency, goal regions,
/// and start positions. Immutable after construction.

#[derive(Debug, Clone)]
pub struct Topology {
    pub board_size: usize,
    pub initial_walls_per_player: u8,
}

impl Topology {
    pub fn new(board_size: usize, initial_walls_per_player: u8) -> Self {
        Self { board_size, initial_walls_per_player }
    }

    pub fn standard() -> Self {
        Self::new(9, 10)
    }

    /// Number of squares per side.
    pub fn n(&self) -> usize {
        self.board_size
    }

    /// Whether (x, y) is a valid square.
    pub fn is_valid_square(&self, x: usize, y: usize) -> bool {
        x < self.n() && y < self.n()
    }

    /// Whether (x, y, horizontal) is a valid horizontal wall head.
    /// Wall occupies edges (x, y, h) and (x+1, y, h).
    /// Needs: x+1 < N and y < N-1.
    pub fn is_valid_h_wall_head(&self, x: usize, y: usize) -> bool {
        let n = self.n();
        x + 1 < n && y + 1 < n
    }

    /// Whether (x, y, vertical) is a valid vertical wall head.
    /// Wall occupies edges (x, y, v) and (x, y+1, v).
    /// Needs: x < N-1 and y+1 < N.
    pub fn is_valid_v_wall_head(&self, x: usize, y: usize) -> bool {
        let n = self.n();
        x + 1 < n && y + 1 < n
    }

    /// Player 1 start: bottom-centre = (N/2, 0).
    pub fn p1_start(&self) -> (usize, usize) {
        (self.n() / 2, 0)
    }

    /// Player 2 start: top-centre = (N/2, N-1).
    pub fn p2_start(&self) -> (usize, usize) {
        (self.n() / 2, self.n() - 1)
    }

    /// Player 1 goal: y = N-1 (top row).
    pub fn p1_goal_y(&self) -> usize {
        self.n() - 1
    }

    /// Player 2 goal: y = 0 (bottom row).
    pub fn p2_goal_y(&self) -> usize {
        0
    }

    /// Goal row for given player.
    pub fn goal_y(&self, player: Player) -> usize {
        match player {
            Player::P1 => self.p1_goal_y(),
            Player::P2 => self.p2_goal_y(),
        }
    }

    /// Start position for given player.
    pub fn start_pos(&self, player: Player) -> (usize, usize) {
        match player {
            Player::P1 => self.p1_start(),
            Player::P2 => self.p2_start(),
        }
    }
}

use crate::model::Player;

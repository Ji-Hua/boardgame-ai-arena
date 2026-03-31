/// Minimal end-to-end wall blockage acceptance test.
///
/// Translated from: quoridor_v0/quoridor-rust-engine/tests-wheel/test_wall_blockage_acceptance.py
/// Coordinates use the new logical system (origin bottom-left, x right, y up).
/// See: engine/tests/test_wall_blockage_minimal_e2e.py for the behavior specification.

use quoridor_engine::error::RuleErrorCode;
use quoridor_engine::model::*;
use quoridor_engine::rule::RuleEngine;

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

// =================================================================
// BLOCK 1 — P2 trapped between two vertical walls
// =================================================================
//
// Setup: P1(4,0) P2(4,7)  V(2,7) V(4,7)  current=P1  p1_rem=8
// P2 goal y=0. Trap at top prevents P2 from reaching y=0.
//
// V(2,7) blocks x=2↔3 at y=7,8
// V(4,7) blocks x=4↔5 at y=7,8
// P2 confined to x∈{3,4} for y∈{7,8}, escape downward through y=6

#[test]
fn block1_h_3_6_illegal_blocks_all_paths() {
    let e = engine();
    let s = custom_state((4, 0), (4, 7), &[], &[(2, 7), (4, 7)], Player::P1, 8, 10);
    let a = Action::place_wall(Player::P1, 3, 6, Orientation::Horizontal);
    let r = e.apply_action(&s, &a);
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, RuleErrorCode::WallBlocksAllPaths);
}

#[test]
fn block1_h_4_6_legal() {
    let e = engine();
    let s = custom_state((4, 0), (4, 7), &[], &[(2, 7), (4, 7)], Player::P1, 8, 10);
    let a = Action::place_wall(Player::P1, 4, 6, Orientation::Horizontal);
    let s2 = e.apply_action(&s, &a).unwrap();
    assert_eq!(s2.current_player, Player::P2);
}

#[test]
fn block1_v_3_7_legal() {
    let e = engine();
    let s = custom_state((4, 0), (4, 7), &[], &[(2, 7), (4, 7)], Player::P1, 8, 10);
    let a = Action::place_wall(Player::P1, 3, 7, Orientation::Vertical);
    let s2 = e.apply_action(&s, &a).unwrap();
    assert_eq!(s2.current_player, Player::P2);
}

#[test]
fn block1_h_2_6_legal() {
    let e = engine();
    let s = custom_state((4, 0), (4, 7), &[], &[(2, 7), (4, 7)], Player::P1, 8, 10);
    let a = Action::place_wall(Player::P1, 2, 6, Orientation::Horizontal);
    let s2 = e.apply_action(&s, &a).unwrap();
    assert_eq!(s2.current_player, Player::P2);
}

// =================================================================
// BLOCK 2 — P2 at different position, same trap
// =================================================================
//
// Setup: P1(4,0) P2(3,7)  V(2,7) V(4,7)  current=P1  p1_rem=8

#[test]
fn block2_h_3_6_illegal_blocks_all_paths() {
    let e = engine();
    let s = custom_state((4, 0), (3, 7), &[], &[(2, 7), (4, 7)], Player::P1, 8, 10);
    let a = Action::place_wall(Player::P1, 3, 6, Orientation::Horizontal);
    let r = e.apply_action(&s, &a);
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, RuleErrorCode::WallBlocksAllPaths);
}

#[test]
fn block2_h_3_7_legal() {
    let e = engine();
    let s = custom_state((4, 0), (3, 7), &[], &[(2, 7), (4, 7)], Player::P1, 8, 10);
    let a = Action::place_wall(Player::P1, 3, 7, Orientation::Horizontal);
    let s2 = e.apply_action(&s, &a).unwrap();
    assert_eq!(s2.current_player, Player::P2);
}

#[test]
fn block2_h_4_6_legal() {
    let e = engine();
    let s = custom_state((4, 0), (3, 7), &[], &[(2, 7), (4, 7)], Player::P1, 8, 10);
    let a = Action::place_wall(Player::P1, 4, 6, Orientation::Horizontal);
    let s2 = e.apply_action(&s, &a).unwrap();
    assert_eq!(s2.current_player, Player::P2);
}

// =================================================================
// BLOCK 3 — P2 trapped between horizontal walls (mirror)
// =================================================================
//
// Setup: P1(0,4) P2(8,4)  H(0,5) H(0,3)  current=P1  p1_rem=8
// P2 goal y=0; trap prevents reaching it.
//
// H(0,5) blocks y=5↔6 at x=0,1
// H(0,3) blocks y=3↔4 at x=0,1
// P2 confined to y∈{4,5} for x∈{0,1}, escape rightward through x=2

#[test]
fn block3_v_1_4_illegal_blocks_all_paths() {
    let e = engine();
    let s = custom_state((0, 4), (8, 4), &[(0, 5), (0, 3)], &[], Player::P1, 8, 10);
    let a = Action::place_wall(Player::P1, 1, 4, Orientation::Vertical);
    let r = e.apply_action(&s, &a);
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, RuleErrorCode::WallBlocksAllPaths);
}

#[test]
fn block3_v_1_3_legal() {
    let e = engine();
    let s = custom_state((0, 4), (8, 4), &[(0, 5), (0, 3)], &[], Player::P1, 8, 10);
    let a = Action::place_wall(Player::P1, 1, 3, Orientation::Vertical);
    let s2 = e.apply_action(&s, &a).unwrap();
    assert_eq!(s2.current_player, Player::P2);
}

#[test]
fn block3_v_1_5_legal() {
    let e = engine();
    let s = custom_state((0, 4), (8, 4), &[(0, 5), (0, 3)], &[], Player::P1, 8, 10);
    let a = Action::place_wall(Player::P1, 1, 5, Orientation::Vertical);
    let s2 = e.apply_action(&s, &a).unwrap();
    assert_eq!(s2.current_player, Player::P2);
}

// =================================================================
// BLOCK 4 — Sequential gameplay with illegal move handling
// =================================================================

#[test]
fn block4_sequential_gameplay() {
    let e = engine();
    let mut s = e.initial_state();
    assert_eq!(s.current_player, Player::P1);
    assert_eq!(s.pawn_pos(Player::P1), (4, 0));
    assert_eq!(s.pawn_pos(Player::P2), (4, 8));

    // 4.1  P1 pawn → (4, 1)  LEGAL
    s = e.apply_action(&s, &Action::move_pawn(Player::P1, 4, 1)).unwrap();
    assert_eq!(s.current_player, Player::P2);
    assert_eq!(s.pawn_pos(Player::P1), (4, 1));

    // 4.2  P2 wall V(3, 0)  LEGAL
    s = e.apply_action(&s, &Action::place_wall(Player::P2, 3, 0, Orientation::Vertical)).unwrap();
    assert_eq!(s.current_player, Player::P1);

    // 4.3  P1 pawn → (4, 2)  LEGAL
    s = e.apply_action(&s, &Action::move_pawn(Player::P1, 4, 2)).unwrap();
    assert_eq!(s.current_player, Player::P2);
    assert_eq!(s.pawn_pos(Player::P1), (4, 2));

    // 4.4  P2 wall V(4, 0)  LEGAL
    s = e.apply_action(&s, &Action::place_wall(Player::P2, 4, 0, Orientation::Vertical)).unwrap();
    assert_eq!(s.current_player, Player::P1);

    // 4.5  P1 pawn → (4, 1)  LEGAL (back down)
    s = e.apply_action(&s, &Action::move_pawn(Player::P1, 4, 1)).unwrap();
    assert_eq!(s.current_player, Player::P2);
    assert_eq!(s.pawn_pos(Player::P1), (4, 1));

    // 4.6.1  P2 wall H(4, 1)  ILLEGAL — WALL_BLOCKS_ALL_PATHS
    let r = e.apply_action(&s, &Action::place_wall(Player::P2, 4, 1, Orientation::Horizontal));
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, RuleErrorCode::WallBlocksAllPaths);
    // State unchanged — still P2's turn
    assert_eq!(s.current_player, Player::P2);

    // 4.6.2  P2 wall H(5, 1)  LEGAL (retry)
    s = e.apply_action(&s, &Action::place_wall(Player::P2, 5, 1, Orientation::Horizontal)).unwrap();
    assert_eq!(s.current_player, Player::P1);
}

// =================================================================
// BLOCK 5 — Sequential wall overlap detection
// =================================================================

#[test]
fn block5_sequential_wall_overlap() {
    let e = engine();
    let mut s = e.initial_state();

    // 5.1  P1 H(0, 0)  LEGAL
    s = e.apply_action(&s, &Action::place_wall(Player::P1, 0, 0, Orientation::Horizontal)).unwrap();
    assert_eq!(s.current_player, Player::P2);

    // 5.2  P2 H(2, 0)  LEGAL
    s = e.apply_action(&s, &Action::place_wall(Player::P2, 2, 0, Orientation::Horizontal)).unwrap();
    assert_eq!(s.current_player, Player::P1);

    // 5.3  P1 H(4, 0)  LEGAL
    s = e.apply_action(&s, &Action::place_wall(Player::P1, 4, 0, Orientation::Horizontal)).unwrap();
    assert_eq!(s.current_player, Player::P2);

    // 5.4  P2 H(6, 0)  LEGAL
    s = e.apply_action(&s, &Action::place_wall(Player::P2, 6, 0, Orientation::Horizontal)).unwrap();
    assert_eq!(s.current_player, Player::P1);

    // 5.5  P1 H(7, 0)  ILLEGAL — WALL_OVERLAP (shares edge (7,0,h) with H(6,0))
    let r = e.apply_action(&s, &Action::place_wall(Player::P1, 7, 0, Orientation::Horizontal));
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, RuleErrorCode::WallOverlap);
    assert_eq!(s.current_player, Player::P1);

    // 5.6  P1 H(7, 1)  LEGAL
    s = e.apply_action(&s, &Action::place_wall(Player::P1, 7, 1, Orientation::Horizontal)).unwrap();
    assert_eq!(s.current_player, Player::P2);

    // 5.7  P2 H(6, 1)  ILLEGAL — WALL_OVERLAP (shares edge (7,1,h) with H(7,1))
    let r = e.apply_action(&s, &Action::place_wall(Player::P2, 6, 1, Orientation::Horizontal));
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, RuleErrorCode::WallOverlap);
    assert_eq!(s.current_player, Player::P2);

    // 5.8  P2 H(5, 1)  LEGAL
    s = e.apply_action(&s, &Action::place_wall(Player::P2, 5, 1, Orientation::Horizontal)).unwrap();
    assert_eq!(s.current_player, Player::P1);

    // 5.9  P1 H(4, 1)  ILLEGAL — WALL_OVERLAP (shares edge (5,1,h) with H(5,1))
    let r = e.apply_action(&s, &Action::place_wall(Player::P1, 4, 1, Orientation::Horizontal));
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().code, RuleErrorCode::WallOverlap);
    assert_eq!(s.current_player, Player::P1);

    // 5.10  P1 H(3, 1)  LEGAL
    s = e.apply_action(&s, &Action::place_wall(Player::P1, 3, 1, Orientation::Horizontal)).unwrap();
    assert_eq!(s.current_player, Player::P2);
}

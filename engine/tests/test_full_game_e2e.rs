/// Full-game end-to-end replay test.
///
/// Parses engine/tests/full_game_replay.md and replays every step
/// against the real engine, asserting:
///   - ACCEPT steps succeed, mutate state, and switch turn
///   - REJECT steps fail, leave state unchanged, and do not switch turn
///   - State snapshots match at documented checkpoints
///   - The game ends with Player 1 winning

use regex::Regex;
use quoridor_engine::model::*;
use quoridor_engine::rule::RuleEngine;

// ─── Step parser ──────────────────────────────────────────────────

#[derive(Debug)]
struct GameStep {
    label: String,
    player: u8,
    kind: String,
    target_x: i32,
    target_y: i32,
    target_type: String,
    accept: bool,
}

fn parse_steps(content: &str) -> Vec<GameStep> {
    let step_re = Regex::new(r"(?m)^Step\s+(\d+(?:\.\d+)?)\s*:").unwrap();
    let action_re = Regex::new(
        r"Action:\s*\{\s*player:\s*(\d+)\s*,\s*kind:\s*(\w+)\s*,\s*target:\s*\((-?\d+)\s*,\s*(-?\d+)\s*,\s*(\w+)\)\s*\}\s*->\s*(ACCEPT|REJECT)"
    ).unwrap();

    let mut steps = Vec::new();
    let mut current_label = String::new();

    for line in content.lines() {
        if let Some(cap) = step_re.captures(line) {
            current_label = cap[1].to_string();
        }
        if let Some(cap) = action_re.captures(line) {
            steps.push(GameStep {
                label: current_label.clone(),
                player: cap[1].parse().unwrap(),
                kind: cap[2].to_string(),
                target_x: cap[3].parse().unwrap(),
                target_y: cap[4].parse().unwrap(),
                target_type: cap[5].to_string(),
                accept: &cap[6] == "ACCEPT",
            });
        }
    }
    steps
}

// ─── Action builder ───────────────────────────────────────────────

fn build_action(step: &GameStep) -> Option<Action> {
    if step.target_x < 0 || step.target_y < 0 || step.target_x >= 9 || step.target_y >= 9 {
        return None;
    }

    let player = match step.player {
        1 => Player::P1,
        2 => Player::P2,
        _ => return None,
    };

    let x = step.target_x as usize;
    let y = step.target_y as usize;

    match step.kind.as_str() {
        "MovePawn" => Some(Action::move_pawn(player, x, y)),
        "PlaceWall" => {
            let orientation = match step.target_type.as_str() {
                "Horizontal" => Orientation::Horizontal,
                "Vertical" => Orientation::Vertical,
                _ => return None,
            };
            Some(Action::place_wall(player, x, y, orientation))
        }
        _ => None,
    }
}

// ─── Snapshot checker ─────────────────────────────────────────────

fn check_snapshot(
    state: &RawState,
    label: &str,
    p1: (usize, usize),
    p2: (usize, usize),
    p1_walls: u8,
    p2_walls: u8,
    failures: &mut Vec<String>,
) {
    if state.pawn_pos(Player::P1) != p1 {
        failures.push(format!(
            "Snapshot after Step {}: P1 at {:?}, expected {:?}",
            label, state.pawn_pos(Player::P1), p1,
        ));
    }
    if state.pawn_pos(Player::P2) != p2 {
        failures.push(format!(
            "Snapshot after Step {}: P2 at {:?}, expected {:?}",
            label, state.pawn_pos(Player::P2), p2,
        ));
    }
    if state.walls_remaining(Player::P1) != p1_walls {
        failures.push(format!(
            "Snapshot after Step {}: P1 walls {}, expected {}",
            label, state.walls_remaining(Player::P1), p1_walls,
        ));
    }
    if state.walls_remaining(Player::P2) != p2_walls {
        failures.push(format!(
            "Snapshot after Step {}: P2 walls {}, expected {}",
            label, state.walls_remaining(Player::P2), p2_walls,
        ));
    }
}

// ─── Test ─────────────────────────────────────────────────────────

#[test]
fn test_full_game_replay() {
    let content = include_str!("../../documents/engine/implementation/full_game_replay.md");
    let steps = parse_steps(content);
    assert_eq!(steps.len(), 109, "Expected 109 steps (61 ACCEPT + 48 REJECT)");

    let engine = RuleEngine::standard();
    let mut state = engine.initial_state();

    assert_eq!(state.pawn_pos(Player::P1), (4, 0));
    assert_eq!(state.pawn_pos(Player::P2), (4, 8));
    assert_eq!(state.current_player, Player::P1);

    let mut accept_count = 0usize;
    let mut reject_count = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for step in &steps {
        let snapshot = state.clone();

        match build_action(step) {
            None => {
                if step.accept {
                    failures.push(format!(
                        "Step {}: unrepresentable action marked ACCEPT (coords {},{},{})",
                        step.label, step.target_x, step.target_y, step.target_type,
                    ));
                } else {
                    reject_count += 1;
                }
            }
            Some(action) => {
                let result = engine.apply_action(&state, &action);

                if step.accept {
                    match result {
                        Ok(new_state) => {
                            if new_state.current_player == snapshot.current_player {
                                failures.push(format!(
                                    "Step {}: turn did not switch on ACCEPT",
                                    step.label,
                                ));
                            }
                            state = new_state;
                            accept_count += 1;

                            // Verify state snapshots at documented checkpoints
                            match step.label.as_str() {
                                "10" => check_snapshot(&state, "10", (4,2), (4,3), 10, 9, &mut failures),
                                "18" => check_snapshot(&state, "18", (3,4), (3,2), 9, 8, &mut failures),
                                "28" => check_snapshot(&state, "28", (3,6), (1,3), 6, 8, &mut failures),
                                "38" => check_snapshot(&state, "38", (5,7), (1,6), 6, 6, &mut failures),
                                "48" => check_snapshot(&state, "48", (4,7), (1,7), 4, 4, &mut failures),
                                "58" => check_snapshot(&state, "58", (2,7), (3,7), 2, 1, &mut failures),
                                _ => {}
                            }
                        }
                        Err(e) => {
                            failures.push(format!(
                                "Step {} expected ACCEPT but got REJECT: {:?}",
                                step.label, e,
                            ));
                        }
                    }
                } else {
                    match result {
                        Ok(_) => {
                            failures.push(format!(
                                "Step {} expected REJECT but got ACCEPT",
                                step.label,
                            ));
                        }
                        Err(_) => {
                            if state != snapshot {
                                failures.push(format!(
                                    "Step {}: state changed on REJECT",
                                    step.label,
                                ));
                            }
                            reject_count += 1;
                        }
                    }
                }
            }
        }
    }

    // Report all failures
    if !failures.is_empty() {
        let msg = failures.join("\n  ");
        panic!(
            "\n{} step failure(s) found:\n  {}\n\n({} ACCEPT ok, {} REJECT ok)",
            failures.len(), msg, accept_count, reject_count,
        );
    }

    // Final assertions
    assert!(engine.is_game_over(&state), "Game should be over after final step");
    assert_eq!(engine.winner(&state), Some(Player::P1), "Player 1 should win");

    eprintln!(
        "Full game replay: {} ACCEPT, {} REJECT, {} total",
        accept_count, reject_count, accept_count + reject_count,
    );
}

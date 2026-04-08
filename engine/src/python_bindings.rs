/// PyO3 bindings for the Quoridor engine.
///
/// Exposes the Rule Engine, state, action, and player types to Python.
/// This module wraps the existing Rust API without modifying it.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::model::{
    Action as RustAction, ActionKind, CoordinateKind,
    Orientation as RustOrientation, Player as RustPlayer, RawState as RustRawState,
};
use crate::rule::RuleEngine as RustRuleEngine;
use crate::topology::Topology as RustTopology;
use crate::calculation;
use crate::error::RuleError as RustRuleError;

// ---------------------------------------------------------------------------
// Player
// ---------------------------------------------------------------------------

#[pyclass(eq, frozen, name = "Player")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyPlayer {
    inner: RustPlayer,
}

#[pymethods]
impl PyPlayer {
    #[classattr]
    const P1: PyPlayer = PyPlayer { inner: RustPlayer::P1 };

    #[classattr]
    const P2: PyPlayer = PyPlayer { inner: RustPlayer::P2 };

    fn __repr__(&self) -> String {
        match self.inner {
            RustPlayer::P1 => "Player.P1".to_string(),
            RustPlayer::P2 => "Player.P2".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self.inner {
            RustPlayer::P1 => "P1".to_string(),
            RustPlayer::P2 => "P2".to_string(),
        }
    }

    fn opponent(&self) -> PyPlayer {
        PyPlayer { inner: self.inner.opponent() }
    }
}

// ---------------------------------------------------------------------------
// Orientation
// ---------------------------------------------------------------------------

#[pyclass(eq, frozen, name = "Orientation")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyOrientation {
    inner: RustOrientation,
}

#[pymethods]
#[allow(non_upper_case_globals)]
impl PyOrientation {
    #[classattr]
    const Horizontal: PyOrientation = PyOrientation { inner: RustOrientation::Horizontal };

    #[classattr]
    const Vertical: PyOrientation = PyOrientation { inner: RustOrientation::Vertical };

    fn __repr__(&self) -> String {
        match self.inner {
            RustOrientation::Horizontal => "Orientation.Horizontal".to_string(),
            RustOrientation::Vertical => "Orientation.Vertical".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Action
// ---------------------------------------------------------------------------

#[pyclass(frozen, name = "Action")]
#[derive(Clone)]
pub struct PyAction {
    inner: RustAction,
}

#[pymethods]
impl PyAction {
    #[staticmethod]
    fn move_pawn(player: PyPlayer, x: usize, y: usize) -> Self {
        PyAction { inner: RustAction::move_pawn(player.inner, x, y) }
    }

    #[staticmethod]
    fn place_wall(player: PyPlayer, x: usize, y: usize, orientation: PyOrientation) -> Self {
        PyAction { inner: RustAction::place_wall(player.inner, x, y, orientation.inner) }
    }

    #[getter]
    fn player(&self) -> PyPlayer {
        PyPlayer { inner: self.inner.player }
    }

    #[getter]
    fn kind(&self) -> String {
        match self.inner.kind {
            ActionKind::MovePawn => "MovePawn".to_string(),
            ActionKind::PlaceWall => "PlaceWall".to_string(),
        }
    }

    #[getter]
    fn target_x(&self) -> usize {
        self.inner.target.x
    }

    #[getter]
    fn target_y(&self) -> usize {
        self.inner.target.y
    }

    #[getter]
    fn coordinate_kind(&self) -> String {
        match self.inner.target.kind {
            CoordinateKind::Square => "Square".to_string(),
            CoordinateKind::Horizontal => "Horizontal".to_string(),
            CoordinateKind::Vertical => "Vertical".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        let kind = match self.inner.kind {
            ActionKind::MovePawn => "MovePawn",
            ActionKind::PlaceWall => "PlaceWall",
        };
        let coord_kind = match self.inner.target.kind {
            CoordinateKind::Square => "Square",
            CoordinateKind::Horizontal => "Horizontal",
            CoordinateKind::Vertical => "Vertical",
        };
        format!(
            "Action({}, {}, ({}, {}, {}))",
            self.inner.player.index() + 1,
            kind,
            self.inner.target.x,
            self.inner.target.y,
            coord_kind,
        )
    }
}

// ---------------------------------------------------------------------------
// RawState
// ---------------------------------------------------------------------------

#[pyclass(frozen, name = "RawState")]
#[derive(Clone)]
pub struct PyRawState {
    inner: RustRawState,
}

#[pymethods]
impl PyRawState {
    #[new]
    #[pyo3(signature = (p1_x, p1_y, p2_x, p2_y, walls_p1, walls_p2, h_edges, v_edges, h_heads, v_heads, current_player))]
    fn new(
        p1_x: usize, p1_y: usize,
        p2_x: usize, p2_y: usize,
        walls_p1: u8, walls_p2: u8,
        h_edges: u128, v_edges: u128,
        h_heads: u128, v_heads: u128,
        current_player: PyPlayer,
    ) -> Self {
        PyRawState {
            inner: RustRawState {
                pawn_positions: [(p1_x, p1_y), (p2_x, p2_y)],
                remaining_walls: [walls_p1, walls_p2],
                horizontal_edges: h_edges,
                vertical_edges: v_edges,
                horizontal_heads: h_heads,
                vertical_heads: v_heads,
                current_player: current_player.inner,
            },
        }
    }

    fn pawn_pos(&self, player: PyPlayer) -> (usize, usize) {
        self.inner.pawn_pos(player.inner)
    }

    fn walls_remaining(&self, player: PyPlayer) -> u8 {
        self.inner.walls_remaining(player.inner)
    }

    #[getter]
    fn current_player(&self) -> PyPlayer {
        PyPlayer { inner: self.inner.current_player }
    }

    #[getter]
    fn horizontal_edges(&self) -> u128 {
        self.inner.horizontal_edges
    }

    #[getter]
    fn vertical_edges(&self) -> u128 {
        self.inner.vertical_edges
    }

    #[getter]
    fn horizontal_heads(&self) -> u128 {
        self.inner.horizontal_heads
    }

    #[getter]
    fn vertical_heads(&self) -> u128 {
        self.inner.vertical_heads
    }

    fn __eq__(&self, other: &PyRawState) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        let (p1x, p1y) = self.inner.pawn_pos(RustPlayer::P1);
        let (p2x, p2y) = self.inner.pawn_pos(RustPlayer::P2);
        format!(
            "RawState(P1=({},{}), P2=({},{}), walls=[{},{}], turn={:?})",
            p1x, p1y, p2x, p2y,
            self.inner.remaining_walls[0],
            self.inner.remaining_walls[1],
            self.inner.current_player,
        )
    }
}

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

#[pyclass(frozen, name = "Topology")]
#[derive(Clone)]
pub struct PyTopology {
    inner: RustTopology,
}

#[pymethods]
impl PyTopology {
    fn n(&self) -> usize {
        self.inner.n()
    }

    fn goal_y(&self, player: PyPlayer) -> usize {
        self.inner.goal_y(player.inner)
    }

    fn start_pos(&self, player: PyPlayer) -> (usize, usize) {
        self.inner.start_pos(player.inner)
    }
}

// ---------------------------------------------------------------------------
// RuleEngine
// ---------------------------------------------------------------------------

#[pyclass(frozen, name = "RuleEngine")]
#[derive(Clone)]
pub struct PyRuleEngine {
    inner: RustRuleEngine,
}

#[pymethods]
impl PyRuleEngine {
    #[staticmethod]
    fn standard() -> Self {
        PyRuleEngine { inner: RustRuleEngine::standard() }
    }

    #[getter]
    fn topology(&self) -> PyTopology {
        PyTopology { inner: self.inner.topology.clone() }
    }

    fn initial_state(&self) -> PyRawState {
        PyRawState { inner: self.inner.initial_state() }
    }

    fn apply_action(&self, state: &PyRawState, action: &PyAction) -> PyResult<PyRawState> {
        self.inner
            .apply_action(&state.inner, &action.inner)
            .map(|s| PyRawState { inner: s })
            .map_err(|e| rule_error_to_py(e))
    }

    fn legal_actions(&self, state: &PyRawState) -> Vec<PyAction> {
        self.inner
            .legal_actions(&state.inner)
            .into_iter()
            .map(|a| PyAction { inner: a })
            .collect()
    }

    fn is_game_over(&self, state: &PyRawState) -> bool {
        self.inner.is_game_over(&state.inner)
    }

    fn winner(&self, state: &PyRawState) -> Option<PyPlayer> {
        self.inner.winner(&state.inner).map(|p| PyPlayer { inner: p })
    }

    fn path_exists(&self, state: &PyRawState, player: PyPlayer) -> bool {
        self.inner.path_exists(&state.inner, player.inner)
    }
}

// ---------------------------------------------------------------------------
// Calculation submodule
// ---------------------------------------------------------------------------

#[pyfunction]
fn shortest_path_len(state: &PyRawState, player: PyPlayer, topology: &PyTopology) -> Option<u32> {
    calculation::shortest_path_len(&state.inner, player.inner, &topology.inner)
}

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

fn rule_error_to_py(e: RustRuleError) -> PyErr {
    PyValueError::new_err(format!("{:?}: {}", e.code, e.message))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
pub fn quoridor_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPlayer>()?;
    m.add_class::<PyOrientation>()?;
    m.add_class::<PyAction>()?;
    m.add_class::<PyRawState>()?;
    m.add_class::<PyTopology>()?;
    m.add_class::<PyRuleEngine>()?;

    // Add calculation submodule
    let calc_module = PyModule::new(m.py(), "calculation")?;
    calc_module.add_function(wrap_pyfunction!(shortest_path_len, &calc_module)?)?;
    m.add_submodule(&calc_module)?;

    Ok(())
}

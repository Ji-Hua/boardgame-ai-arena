/// Rule error codes matching the interface specification.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuleErrorCode {
    InvalidPlayer,
    WrongTurn,
    InvalidActionKind,
    GameOver,
    PawnMoveOutOfBounds,
    WallOutOfBounds,
    PawnMoveBlocked,
    PawnMoveSameSquare,
    PawnMoveOccupied,
    PawnMoveNotAdjacent,
    WallOverlap,
    WallCrossing,
    NoWallsRemaining,
    WallBlocksAllPaths,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleError {
    pub code: RuleErrorCode,
    pub message: String,
}

impl RuleError {
    pub fn new(code: RuleErrorCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }
}

pub type RuleResult<T> = Result<T, RuleError>;

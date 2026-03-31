pub mod rule_engine;
pub mod game_manager;

// Re-export rule_engine submodules at crate root for backward compatibility.
pub use rule_engine::topology;
pub use rule_engine::model;
pub use rule_engine::rule;
pub use rule_engine::calculation;
pub use rule_engine::error;

#[cfg(feature = "python")]
mod python_bindings;

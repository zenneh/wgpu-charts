//! Core types for technical analysis.

pub mod direction;
pub mod level;
pub mod range;

pub use direction::{CandleDirection, CandleMetadata};
pub use level::{Level, LevelBreak, LevelEvent, LevelHit, LevelId, LevelInteraction, LevelState, LevelTracker, LevelType};
pub use range::{detect_ranges, Range, RangeBuilder, RangeId};

//! Core types for technical analysis.

pub mod direction;
pub mod level;
pub mod range;
pub mod trend;

pub use direction::{CandleDirection, CandleMetadata};
pub use level::{BucketKey, Level, LevelBreak, LevelEvent, LevelHit, LevelId, LevelInteraction, LevelState, LevelTracker, LevelType, OptimizedLevelTracker};
pub use range::{detect_ranges, Range, RangeBuilder, RangeId};
pub use trend::{Trend, TrendBreak, TrendEvent, TrendHit, TrendId, TrendInteraction, TrendState, TrendTracker};

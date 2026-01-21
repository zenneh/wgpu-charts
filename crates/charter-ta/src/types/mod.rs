//! Core types for technical analysis.

pub mod candle;
pub mod config;
pub mod level;
pub mod range;
pub mod trend;

pub use candle::{CandleDirection, CandleMetadata};
pub use config::{AnalyzerConfig, TimeframeConfig};
pub use level::{
    Level, LevelBreak, LevelDirection, LevelEvent, LevelHit, LevelId, LevelIndex,
    LevelInteraction, LevelState, LevelType,
};
pub use range::{detect_ranges, detect_ranges_reverse, Range, RangeBuilder, RangeId};
pub use trend::{
    Trend, TrendBreak, TrendEvent, TrendHit, TrendId, TrendInteraction, TrendPoint,
    TrendState, TrendTracker,
};

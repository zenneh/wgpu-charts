//! Charter Technical Analysis - Rule-based price action analysis.
//!
//! This crate provides a high-performance technical analysis framework using
//! a two-pass algorithm (reverse + forward) with formal guarantees for
//! early-stopping optimization.
//!
//! # Core Concepts
//!
//! - **Candles**: OHLCV data points with computed direction (bullish/bearish/doji)
//! - **Ranges**: Consecutive candles of the same direction
//! - **Levels**: Price levels derived from ranges (hold and greedy hold levels)
//!
//! # Architecture
//!
//! The analyzer uses a two-pass algorithm:
//!
//! 1. **Reverse Pass**: Process candles from newest to oldest to detect ranges
//!    and levels in O(K) time with early stopping
//! 2. **Forward Pass**: Evaluate level interactions (hits/breaks)
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use charter_ta::{DefaultAnalyzer, AnalyzerConfig, TimeframeConfig, Analyzer};
//! use charter_core::{Candle, Timeframe};
//!
//! // Configure timeframes
//! let config = AnalyzerConfig::new(vec![
//!     TimeframeConfig::new(Timeframe::Hour1, 3, 0.001),
//! ]);
//!
//! // Create analyzer
//! let mut analyzer = DefaultAnalyzer::new(config);
//!
//! // Process candles
//! let candles = vec![/* your candle data */];
//! let result = analyzer.update(0, &candles, 105.0);
//!
//! // Check results
//! for range in &result.ranges {
//!     println!("Range: {:?}", range.direction);
//! }
//! for level in &result.levels {
//!     println!("Level at {}", level.price);
//! }
//! ```

pub mod analyzer;
pub mod ml;
pub mod types;

// Re-export main analyzer types
pub use analyzer::{
    aggregate_candles, AnalysisResult, Analyzer, AnalyzerState, DefaultAnalyzer,
    ForwardPassResult, MultiTimeframeAnalyzer, MultiTimeframeResult, ReversePassResult,
    TimeframeData, TimeframeDataBuilder, TimeframeState,
};

// Re-export core types
pub use types::{
    AnalyzerConfig, CandleDirection, CandleMetadata, Level, LevelBreak, LevelDirection,
    LevelEvent, LevelHit, LevelId, LevelIndex, LevelInteraction, LevelState, LevelType, Range,
    RangeBuilder, RangeId, TimeframeConfig, detect_ranges, detect_ranges_reverse,
};

// Re-export ML types
pub use ml::{
    ExtractionError, ExtractionRequirements, FeatureExtractor, LevelFeatures, MlFeatures,
    TimeframeFeatures, N_LEVELS,
};

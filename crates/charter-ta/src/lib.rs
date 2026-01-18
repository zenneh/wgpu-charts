//! Charter Technical Analysis - Rule-based price action analysis.
//!
//! This crate provides a rule-based framework for analyzing price action
//! through the concepts of:
//!
//! - **Candles**: OHLCV data points with computed direction (bullish/bearish/doji)
//! - **Ranges**: Consecutive candles of the same direction
//! - **Levels**: Price levels derived from ranges (hold and greedy hold levels)
//! - **Rules**: Composable conditions for pattern detection
//!
//! # Quick Start
//!
//! ```rust
//! use charter_ta::{Analyzer, AnalyzerConfig};
//! use charter_core::Candle;
//!
//! // Create an analyzer
//! let mut analyzer = Analyzer::builder()
//!     .min_range_candles(2)
//!     .level_tolerance(0.5)
//!     .build();
//!
//! // Process candles
//! let candle = Candle::new(0.0, 100.0, 110.0, 95.0, 105.0, 1000.0);
//! let result = analyzer.process_candle(candle);
//!
//! // Check for completed ranges
//! for range in &result.completed_ranges {
//!     println!("New {} range with {} candles", range.direction, range.candle_count);
//! }
//!
//! // Check for level events
//! for event in &result.level_events {
//!     println!("Level event: {:?}", event);
//! }
//! ```
//!
//! # Concepts
//!
//! ## Ranges
//!
//! A range is formed when multiple candles of the same direction appear
//! consecutively. For example, 3 bullish candles in a row form a single
//! bullish range.
//!
//! ## Levels
//!
//! Levels are derived from completed ranges:
//!
//! - **Bearish ranges** create levels at the highs:
//!   - Hold level: `max(first_candle_high, last_candle_high)`
//!   - Greedy hold: `min(first_candle_high, last_candle_high)`
//!
//! - **Bullish ranges** create levels at the lows:
//!   - Hold level: `min(first_candle_low, last_candle_low)`
//!   - Greedy hold: `max(first_candle_low, last_candle_low)`
//!
//! ## Level Interactions
//!
//! Levels track how price interacts with them:
//!
//! - **Bearish level hit**: The wick touches or goes below the level,
//!   but the body closes above.
//! - **Bullish level hit**: The wick touches or goes above the level,
//!   but the body closes below.
//! - **Broken**: The full body (both open and close) moves through the level.
//!
//! # Rules Engine
//!
//! The rules engine allows you to define composable conditions:
//!
//! ```rust
//! use charter_ta::rules::{IsBullish, IsBearish, ConsecutiveBearish, Rule};
//!
//! // Simple rule
//! let bullish = IsBullish;
//!
//! // Combined rules
//! let bearish_reversal = ConsecutiveBearish::new(3).and(IsBullish);
//!
//! // Rules can be evaluated on a context
//! // let result = bullish.evaluate(&ctx);
//! ```
//!
//! # Performance
//!
//! The analyzer is designed for streaming data:
//!
//! - Incremental processing via `process_candle()`
//! - Pre-computed candle metadata (direction, body ratio, wicks)
//! - Efficient range detection with minimal allocations
//!

pub mod analyzer;
pub mod rules;
pub mod types;

// Re-export main types
pub use analyzer::{Analyzer, AnalyzerBuilder, AnalyzerConfig, AnalysisResult};
pub use types::{
    CandleDirection, CandleMetadata, Level, LevelEvent, LevelHit, LevelId, LevelState, LevelType,
    Range, RangeId, Trend, TrendEvent, TrendHit, TrendId, TrendState,
};

// Re-export commonly used rules
pub use rules::{Rule, RuleContext};

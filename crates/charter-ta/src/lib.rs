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
pub mod export;
pub mod inference;
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
    // Trend types
    Trend, TrendBreak, TrendEvent, TrendHit, TrendId, TrendInteraction, TrendPoint,
    TrendState, TrendTracker,
};

// Re-export ML types
pub use ml::{
    ExtractionError, ExtractionRequirements, FeatureExtractor, LevelFeatures, MlFeatures,
    MlPrediction, TimeframeFeatures, N_LEVELS,
    // Level event types for hold/break prediction
    LevelApproachEvent, LevelEventFeatures, LevelOutcome,
    determine_outcome, extract_level_features, is_approaching_level,
    APPROACH_THRESHOLD, BREAK_THRESHOLD, HOLD_THRESHOLD,
};

// Re-export inference types
pub use inference::{MlInference, MlInferenceHandle, ScalerParams};

// Re-export export types
pub use export::{
    CsvExporter, ExportConfig, ExportError, Exporter, FeatureRow, LabelMode,
    export_features_csv, export_features_csv_with_config,
};

use charter_core::Timeframe;

/// Returns the default timeframes used for ML feature export.
/// These are the 7 timeframes: 1m, 3m, 5m, 30m, 1h, 1d, 1w
pub fn ml_export_timeframes() -> Vec<Timeframe> {
    vec![
        Timeframe::Min1,
        Timeframe::Min3,
        Timeframe::Min5,
        Timeframe::Min30,
        Timeframe::Hour1,
        Timeframe::Day1,
        Timeframe::Week1,
    ]
}

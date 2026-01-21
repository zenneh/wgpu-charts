//! ML Feature extraction for technical analysis.
//!
//! This module provides trait definitions and implementations for extracting
//! machine learning features from analyzer state.

mod features;
pub mod level_events;

pub use features::{
    compute_momentum, compute_price_action, extract_features_from_state,
    extract_features_with_candles, ExtractionError, ExtractionRequirements, FeatureExtractor,
    LevelFeatures, MlFeatures, MlPrediction, TimeframeFeatures, N_LEVELS,
};

pub use level_events::{
    determine_outcome, extract_level_features, is_approaching_level, LevelApproachEvent,
    LevelEventFeatures, LevelOutcome, APPROACH_THRESHOLD, BREAK_THRESHOLD, HOLD_THRESHOLD,
};

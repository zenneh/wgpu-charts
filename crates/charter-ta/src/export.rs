//! Feature export module for ML training data generation.
//!
//! This module provides traits and implementations for exporting technical analysis
//! features to files for machine learning model training.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use charter_core::{Candle, Timeframe};

use crate::analyzer::{aggregate_candles, Analyzer, DefaultAnalyzer};
use crate::ml::{compute_momentum, compute_price_action, extract_features_from_state, MlFeatures, N_LEVELS};
use crate::types::{AnalyzerConfig, TimeframeConfig};

/// Errors that can occur during export.
#[derive(Debug)]
pub enum ExportError {
    /// IO error during file operations.
    Io(std::io::Error),
    /// Not enough data for the requested timeframes.
    InsufficientData(String),
    /// Invalid configuration.
    InvalidConfig(String),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::Io(e) => write!(f, "IO error: {}", e),
            ExportError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            ExportError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
        }
    }
}

impl std::error::Error for ExportError {}

impl From<std::io::Error> for ExportError {
    fn from(e: std::io::Error) -> Self {
        ExportError::Io(e)
    }
}

/// How to label price direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LabelMode {
    /// Binary labels: UP (1) or DOWN (0). No threshold, any movement counts.
    Binary,
    /// Binary with threshold: only label if move exceeds threshold, skip neutral.
    BinaryFiltered,
    /// Ternary labels: UP (1), DOWN (-1), NEUTRAL (0).
    Ternary,
}

impl Default for LabelMode {
    fn default() -> Self {
        LabelMode::BinaryFiltered
    }
}

/// Configuration for feature export.
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// Timeframes to analyze and export features for.
    pub timeframes: Vec<Timeframe>,
    /// Minimum candles required per timeframe before extracting features.
    pub min_candles_per_timeframe: HashMap<Timeframe, usize>,
    /// Minimum ranges required per timeframe.
    pub min_ranges_per_timeframe: HashMap<Timeframe, usize>,
    /// Minimum active levels required across all timeframes.
    pub min_active_levels: usize,
    /// Minimum range candles for range detection.
    pub min_range_candles: usize,
    /// Doji threshold for candle direction.
    pub doji_threshold: f32,
    /// Whether to create greedy levels.
    pub create_greedy_levels: bool,
    /// Age normalization factor for level features.
    pub age_normalization: f32,
    /// Lookahead candles for direction label (how many candles to look ahead for price direction).
    pub label_lookahead: usize,
    /// Use numbered feature columns (f0, f1, ...) for ML training compatibility.
    pub use_numbered_features: bool,
    /// Minimum price movement threshold for labeling (e.g., 0.001 = 0.1%).
    /// Moves smaller than this are considered neutral.
    pub label_threshold: f32,
    /// How to handle labeling (binary, filtered, ternary).
    pub label_mode: LabelMode,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            timeframes: vec![
                Timeframe::Min1,
                Timeframe::Min3,
                Timeframe::Min5,
                Timeframe::Min30,
                Timeframe::Hour1,
                Timeframe::Day1,
                Timeframe::Week1,
            ],
            min_candles_per_timeframe: HashMap::new(),
            min_ranges_per_timeframe: HashMap::new(),
            min_active_levels: 1,
            min_range_candles: 3,
            doji_threshold: 0.001,
            create_greedy_levels: true,
            age_normalization: 1000.0,
            label_lookahead: 2,      // Reduced from 5 for short-term predictions
            use_numbered_features: true,
            label_threshold: 0.005,  // 0.5% threshold (increased from 0.1% to capture real moves)
            label_mode: LabelMode::BinaryFiltered,
        }
    }
}

impl ExportConfig {
    /// Create a new export config with default timeframes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the timeframes to analyze.
    pub fn with_timeframes(mut self, timeframes: Vec<Timeframe>) -> Self {
        self.timeframes = timeframes;
        self
    }

    /// Set minimum candles for a specific timeframe.
    pub fn with_min_candles(mut self, tf: Timeframe, count: usize) -> Self {
        self.min_candles_per_timeframe.insert(tf, count);
        self
    }

    /// Set minimum ranges for a specific timeframe.
    pub fn with_min_ranges(mut self, tf: Timeframe, count: usize) -> Self {
        self.min_ranges_per_timeframe.insert(tf, count);
        self
    }

    /// Set minimum active levels.
    pub fn with_min_active_levels(mut self, count: usize) -> Self {
        self.min_active_levels = count;
        self
    }

    /// Set minimum range candles.
    pub fn with_min_range_candles(mut self, count: usize) -> Self {
        self.min_range_candles = count;
        self
    }

    /// Set doji threshold.
    pub fn with_doji_threshold(mut self, threshold: f32) -> Self {
        self.doji_threshold = threshold;
        self
    }

    /// Get the timeframe index in the configured timeframes list.
    pub fn timeframe_index(&self, tf: Timeframe) -> Option<u8> {
        self.timeframes.iter().position(|&t| t == tf).map(|i| i as u8)
    }

    /// Set lookahead candles for direction label.
    pub fn with_label_lookahead(mut self, candles: usize) -> Self {
        self.label_lookahead = candles;
        self
    }

    /// Set whether to use numbered feature columns.
    pub fn with_numbered_features(mut self, numbered: bool) -> Self {
        self.use_numbered_features = numbered;
        self
    }

    /// Set the label threshold (minimum price movement to be considered UP or DOWN).
    /// E.g., 0.001 = 0.1%, 0.005 = 0.5%
    pub fn with_label_threshold(mut self, threshold: f32) -> Self {
        self.label_threshold = threshold;
        self
    }

    /// Set the label mode (Binary, BinaryFiltered, or Ternary).
    pub fn with_label_mode(mut self, mode: LabelMode) -> Self {
        self.label_mode = mode;
        self
    }
}

/// Trait for exporting features to various formats.
pub trait Exporter {
    /// Export features from 1-minute candles to a file.
    ///
    /// # Arguments
    /// * `candles_1m` - Base 1-minute candle data
    /// * `output_path` - Path to write the output file
    /// * `config` - Export configuration
    ///
    /// # Returns
    /// Number of feature rows exported
    fn export(
        &self,
        candles_1m: &[Candle],
        output_path: &Path,
        config: &ExportConfig,
    ) -> Result<usize, ExportError>;

    /// Get the file extension for this exporter format.
    fn extension(&self) -> &'static str;
}

/// Result of a single feature extraction at a point in time.
#[derive(Debug, Clone)]
pub struct FeatureRow {
    /// Timestamp of the candle this feature was extracted at.
    pub timestamp: f64,
    /// Current price at extraction time.
    pub current_price: f32,
    /// The extracted ML features.
    pub features: MlFeatures,
}

/// CSV exporter for ML training data.
pub struct CsvExporter;

impl CsvExporter {
    /// Create a new CSV exporter.
    pub fn new() -> Self {
        Self
    }

    /// Calculate the number of features for a given config.
    fn feature_count(config: &ExportConfig) -> usize {
        // Global features: 15 (8 original + 4 price action + 3 momentum)
        // Per-timeframe: 3 + (N_LEVELS * 6) * 2
        let per_timeframe = 3 + (N_LEVELS * 6) * 2;
        15 + config.timeframes.len() * per_timeframe
    }

    /// Generate CSV header row.
    fn generate_header(config: &ExportConfig) -> String {
        if config.use_numbered_features {
            // Numbered features for ML training: f0, f1, f2, ... + direction_up
            let num_features = Self::feature_count(config);
            let mut headers: Vec<String> = (0..num_features).map(|i| format!("f{}", i)).collect();
            headers.push("direction_up".to_string());
            headers.join(",")
        } else {
            // Named features for human readability
            let mut headers = vec![
                "timestamp".to_string(),
                "current_price".to_string(),
                "reference_price".to_string(),
                "total_active_levels".to_string(),
                "total_levels".to_string(),
                "has_resistance_above".to_string(),
                "has_support_below".to_string(),
                "closest_resistance_distance".to_string(),
                "closest_support_distance".to_string(),
            ];

            // Per-timeframe headers
            for (tf_idx, tf) in config.timeframes.iter().enumerate() {
                let prefix = format!("tf{}_{}", tf_idx, tf.label());
                headers.push(format!("{}_active_levels", prefix));
                headers.push(format!("{}_total_levels", prefix));
                headers.push(format!("{}_range_count", prefix));

                // Support levels
                for i in 0..N_LEVELS {
                    headers.push(format!("{}_sup{}_exists", prefix, i));
                    headers.push(format!("{}_sup{}_distance", prefix, i));
                    headers.push(format!("{}_sup{}_hits", prefix, i));
                    headers.push(format!("{}_sup{}_respected_ratio", prefix, i));
                    headers.push(format!("{}_sup{}_is_active", prefix, i));
                    headers.push(format!("{}_sup{}_age", prefix, i));
                }

                // Resistance levels
                for i in 0..N_LEVELS {
                    headers.push(format!("{}_res{}_exists", prefix, i));
                    headers.push(format!("{}_res{}_distance", prefix, i));
                    headers.push(format!("{}_res{}_hits", prefix, i));
                    headers.push(format!("{}_res{}_respected_ratio", prefix, i));
                    headers.push(format!("{}_res{}_is_active", prefix, i));
                    headers.push(format!("{}_res{}_age", prefix, i));
                }
            }

            headers.push("direction_up".to_string());
            headers.join(",")
        }
    }

    /// Format a feature row as CSV.
    fn format_row(row: &FeatureRow, label: i8, label_mode: LabelMode) -> String {
        let mut values = Vec::new();

        // Global features (15 total)
        // Original 8 features
        values.push(format!("{}", row.features.current_price));
        values.push(format!("{}", row.features.reference_price));
        values.push(format!("{}", row.features.total_active_levels));
        values.push(format!("{}", row.features.total_levels));
        values.push(format!("{}", if row.features.has_resistance_above { 1.0 } else { 0.0 }));
        values.push(format!("{}", if row.features.has_support_below { 1.0 } else { 0.0 }));
        values.push(format!("{}", row.features.closest_resistance_distance.unwrap_or(0.0)));
        values.push(format!("{}", row.features.closest_support_distance.unwrap_or(0.0)));

        // Price action features (4)
        values.push(format!("{}", row.features.body_ratio));
        values.push(format!("{}", row.features.upper_wick_ratio));
        values.push(format!("{}", row.features.lower_wick_ratio));
        values.push(format!("{}", row.features.is_bullish));

        // Momentum features (3)
        values.push(format!("{}", row.features.price_change_1));
        values.push(format!("{}", row.features.price_change_3));
        values.push(format!("{}", row.features.price_change_5));

        // Per-timeframe features
        for tf in &row.features.timeframes {
            values.push(format!("{}", tf.active_level_count));
            values.push(format!("{}", tf.total_level_count));
            values.push(format!("{}", tf.range_count));

            // Support levels
            for level in &tf.support_levels {
                values.push(format!("{}", level.exists));
                values.push(format!("{}", level.price_distance));
                values.push(format!("{}", level.hit_count));
                values.push(format!("{}", level.respected_ratio));
                values.push(format!("{}", level.is_active));
                values.push(format!("{}", level.age_normalized));
            }

            // Resistance levels
            for level in &tf.resistance_levels {
                values.push(format!("{}", level.exists));
                values.push(format!("{}", level.price_distance));
                values.push(format!("{}", level.hit_count));
                values.push(format!("{}", level.respected_ratio));
                values.push(format!("{}", level.is_active));
                values.push(format!("{}", level.age_normalized));
            }
        }

        // Direction label
        // For binary modes: 1 = UP, 0 = DOWN
        // For ternary mode: 1 = UP, 0 = NEUTRAL, -1 = DOWN
        // But for ML training, we use direction_up (1 or 0) for binary classification
        let label_value = match label_mode {
            LabelMode::Ternary => label as i32, // -1, 0, 1
            _ => if label == 1 { 1 } else { 0 }, // Binary: 1 or 0
        };
        values.push(format!("{}", label_value));

        values.join(",")
    }
}

impl Default for CsvExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Exporter for CsvExporter {
    fn export(
        &self,
        candles_1m: &[Candle],
        output_path: &Path,
        config: &ExportConfig,
    ) -> Result<usize, ExportError> {
        if candles_1m.is_empty() {
            return Err(ExportError::InsufficientData(
                "No candles provided".to_string(),
            ));
        }

        if config.timeframes.is_empty() {
            return Err(ExportError::InvalidConfig(
                "No timeframes configured".to_string(),
            ));
        }

        // Check if we have enough candles for minimum requirements
        for &tf in &config.timeframes {
            let min_required = config
                .min_candles_per_timeframe
                .get(&tf)
                .copied()
                .unwrap_or(10);
            if candles_1m.len() < min_required {
                return Err(ExportError::InsufficientData(format!(
                    "Need at least {} candles for {:?}, got {}",
                    min_required,
                    tf,
                    candles_1m.len()
                )));
            }
        }

        // Pre-aggregate candles for all timeframes
        let mut timeframe_candles: Vec<Vec<Candle>> = Vec::new();
        for &tf in &config.timeframes {
            let candles = if tf == Timeframe::Min1 {
                candles_1m.to_vec()
            } else {
                aggregate_candles(candles_1m, tf)
            };
            timeframe_candles.push(candles);
        }

        // Create analyzer configs for each timeframe
        let tf_configs: Vec<TimeframeConfig> = config
            .timeframes
            .iter()
            .map(|&tf| {
                let mut tf_config =
                    TimeframeConfig::new(tf, config.min_range_candles, config.doji_threshold);
                if !config.create_greedy_levels {
                    tf_config = tf_config.without_greedy_levels();
                }
                tf_config
            })
            .collect();

        let analyzer_config = AnalyzerConfig::new(tf_configs);

        // Create a single analyzer that we'll update incrementally
        let mut analyzer = DefaultAnalyzer::new(analyzer_config);

        // Open output file
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writeln!(writer, "{}", Self::generate_header(config))?;

        let mut rows_written = 0;
        let total_candles = candles_1m.len();

        // Pre-compute mapping: for each 1m candle index, how many higher TF candles are complete
        // This avoids O(n) scan per candle in the main loop
        let mut tf_counts_at_1m: Vec<Vec<usize>> = vec![Vec::with_capacity(total_candles); config.timeframes.len()];
        for (tf_idx, &tf) in config.timeframes.iter().enumerate() {
            if tf == Timeframe::Min1 {
                // For 1m, it's just 1, 2, 3, ... n
                for i in 0..total_candles {
                    tf_counts_at_1m[tf_idx].push(i + 1);
                }
            } else {
                // For higher TFs, track which bucket each 1m candle falls into
                let mut count = 0;
                let mut last_bucket: Option<i64> = None;
                for candle in candles_1m.iter() {
                    let bucket = (candle.timestamp / tf.seconds()).floor() as i64;
                    if last_bucket.is_none() || bucket != last_bucket.unwrap() {
                        if last_bucket.is_some() {
                            count += 1; // Previous bucket completed
                        }
                        last_bucket = Some(bucket);
                    }
                    tf_counts_at_1m[tf_idx].push(count);
                }
            }
        }

        // Process candles incrementally - only update analyzer when new TF candles complete
        let mut last_tf_counts: Vec<usize> = vec![0; config.timeframes.len()];

        // Class distribution counters
        let mut up_count: usize = 0;
        let mut down_count: usize = 0;
        let mut neutral_count: usize = 0;
        let mut skipped_neutral: usize = 0;

        for end_idx in 1..=total_candles {
            let current_candle = &candles_1m[end_idx - 1];
            let current_price = current_candle.close;

            // Check if we have enough lookahead data for the direction label
            let lookahead_idx = end_idx + config.label_lookahead;
            if lookahead_idx > total_candles {
                continue; // Not enough future data for label
            }

            // Check if we have enough data for all timeframes and update analyzer
            let mut has_enough_data = true;

            for (tf_idx, &tf) in config.timeframes.iter().enumerate() {
                // Get precomputed count of complete candles for this timeframe
                let tf_candle_count = tf_counts_at_1m[tf_idx][end_idx - 1];

                let min_required = config
                    .min_candles_per_timeframe
                    .get(&tf)
                    .copied()
                    .unwrap_or(10);

                if tf_candle_count < min_required {
                    has_enough_data = false;
                    break;
                }

                // Update analyzer if this timeframe has new candles
                if tf_candle_count > last_tf_counts[tf_idx] {
                    let tf_candles_slice = &timeframe_candles[tf_idx][..tf_candle_count];
                    analyzer.update(tf_idx as u8, tf_candles_slice, current_price);
                    last_tf_counts[tf_idx] = tf_candle_count;
                }
            }

            if !has_enough_data {
                continue;
            }

            // Check extraction requirements
            let state = analyzer.state();

            // Check minimum active levels
            if state.total_active_levels() < config.min_active_levels {
                continue;
            }

            // Check minimum ranges per timeframe
            let mut ranges_ok = true;
            for (tf_idx, &tf) in config.timeframes.iter().enumerate() {
                if let Some(&min_ranges) = config.min_ranges_per_timeframe.get(&tf) {
                    if let Some(tf_state) = state.get_timeframe(tf_idx as u8) {
                        if tf_state.range_count() < min_ranges {
                            ranges_ok = false;
                            break;
                        }
                    } else {
                        ranges_ok = false;
                        break;
                    }
                }
            }

            if !ranges_ok {
                continue;
            }

            // Calculate direction label with threshold
            let future_price = candles_1m[lookahead_idx - 1].close;
            let price_change = (future_price - current_price) / current_price;

            // Determine label based on mode and threshold
            let label: Option<i8> = match config.label_mode {
                LabelMode::Binary => {
                    // Original behavior: any movement counts
                    Some(if future_price > current_price { 1 } else { 0 })
                }
                LabelMode::BinaryFiltered => {
                    // Only label if movement exceeds threshold, skip neutral
                    if price_change > config.label_threshold {
                        Some(1) // UP
                    } else if price_change < -config.label_threshold {
                        Some(0) // DOWN
                    } else {
                        None // Skip neutral (within threshold)
                    }
                }
                LabelMode::Ternary => {
                    // Three classes: UP (1), NEUTRAL (0), DOWN (-1)
                    if price_change > config.label_threshold {
                        Some(1)
                    } else if price_change < -config.label_threshold {
                        Some(-1)
                    } else {
                        Some(0)
                    }
                }
            };

            // Skip if no label (filtered mode with neutral movement)
            let label = match label {
                Some(l) => l,
                None => {
                    skipped_neutral += 1;
                    continue;
                }
            };

            // Track class distribution
            match label {
                1 => up_count += 1,
                0 => {
                    if config.label_mode == LabelMode::Ternary {
                        neutral_count += 1;
                    } else {
                        down_count += 1;
                    }
                }
                -1 => down_count += 1,
                _ => {}
            }

            // Extract features
            let mut features = extract_features_from_state(state, end_idx, config.age_normalization);

            // Validate we have features for all timeframes
            if features.timeframes.len() != config.timeframes.len() {
                continue;
            }

            // Compute price action features from current candle
            let (body_ratio, upper_wick_ratio, lower_wick_ratio, is_bullish) =
                compute_price_action(current_candle);
            features.body_ratio = body_ratio;
            features.upper_wick_ratio = upper_wick_ratio;
            features.lower_wick_ratio = lower_wick_ratio;
            features.is_bullish = is_bullish;

            // Compute momentum features from recent candles
            // Need at least 6 candles for full momentum calculation
            let momentum_start = end_idx.saturating_sub(6);
            let momentum_candles = &candles_1m[momentum_start..end_idx];
            let (pc1, pc3, pc5) = compute_momentum(momentum_candles);
            features.price_change_1 = pc1;
            features.price_change_3 = pc3;
            features.price_change_5 = pc5;

            // Create and write the row
            let row = FeatureRow {
                timestamp: current_candle.timestamp,
                current_price,
                features,
            };

            writeln!(writer, "{}", Self::format_row(&row, label, config.label_mode))?;
            rows_written += 1;

            // Progress reporting
            if end_idx % 10000 == 0 {
                eprintln!("  Progress: {}/{} candles ({:.1}%), {} rows exported",
                    end_idx, total_candles,
                    (end_idx as f64 / total_candles as f64) * 100.0,
                    rows_written);
            }
        }

        // Report class distribution
        eprintln!("\n📊 Class Distribution:");
        if config.label_mode == LabelMode::Ternary {
            let total = up_count + down_count + neutral_count;
            eprintln!("   UP:      {} ({:.1}%)", up_count, 100.0 * up_count as f64 / total as f64);
            eprintln!("   NEUTRAL: {} ({:.1}%)", neutral_count, 100.0 * neutral_count as f64 / total as f64);
            eprintln!("   DOWN:    {} ({:.1}%)", down_count, 100.0 * down_count as f64 / total as f64);
        } else {
            let total = up_count + down_count;
            if total > 0 {
                eprintln!("   UP:   {} ({:.1}%)", up_count, 100.0 * up_count as f64 / total as f64);
                eprintln!("   DOWN: {} ({:.1}%)", down_count, 100.0 * down_count as f64 / total as f64);
            }
        }
        if skipped_neutral > 0 {
            eprintln!("   Skipped (neutral): {}", skipped_neutral);
        }

        writer.flush()?;

        Ok(rows_written)
    }

    fn extension(&self) -> &'static str {
        "csv"
    }
}

/// Export features using the default configuration.
pub fn export_features_csv(
    candles_1m: &[Candle],
    output_path: &Path,
) -> Result<usize, ExportError> {
    let exporter = CsvExporter::new();
    let config = ExportConfig::default();
    exporter.export(candles_1m, output_path, &config)
}

/// Export features with custom configuration.
pub fn export_features_csv_with_config(
    candles_1m: &[Candle],
    output_path: &Path,
    config: &ExportConfig,
) -> Result<usize, ExportError> {
    let exporter = CsvExporter::new();
    exporter.export(candles_1m, output_path, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn make_candle(timestamp: f64, open: f32, high: f32, low: f32, close: f32) -> Candle {
        Candle::new(timestamp, open, high, low, close, 100.0)
    }

    fn generate_test_candles(count: usize) -> Vec<Candle> {
        let mut candles = Vec::with_capacity(count);
        let mut price = 100.0f32;

        for i in 0..count {
            let timestamp = (i as f64) * 60.0; // 1 minute intervals
            let change = ((i % 10) as f32 - 5.0) * 0.1; // Oscillating price
            let open = price;
            price += change;
            let close = price;
            let high = open.max(close) + 0.5;
            let low = open.min(close) - 0.5;

            candles.push(make_candle(timestamp, open, high, low, close));
        }

        candles
    }

    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert_eq!(config.timeframes.len(), 7);
        assert!(config.timeframes.contains(&Timeframe::Min1));
        assert!(config.timeframes.contains(&Timeframe::Week1));
    }

    #[test]
    fn test_csv_header_generation_numbered() {
        let config = ExportConfig::default()
            .with_timeframes(vec![Timeframe::Min1, Timeframe::Min5])
            .with_numbered_features(true);
        let header = CsvExporter::generate_header(&config);

        // Numbered features: f0, f1, f2, ... + direction_up
        assert!(header.starts_with("f0,"));
        assert!(header.contains("direction_up"));
        assert!(!header.contains("timestamp")); // No named columns in numbered mode
    }

    #[test]
    fn test_csv_header_generation_named() {
        let config = ExportConfig::default()
            .with_timeframes(vec![Timeframe::Min1, Timeframe::Min5])
            .with_numbered_features(false);
        let header = CsvExporter::generate_header(&config);

        assert!(header.contains("current_price"));
        assert!(header.contains("tf0_1m_active_levels"));
        assert!(header.contains("tf1_5m_res0_exists"));
        assert!(header.contains("direction_up"));
    }

    #[test]
    fn test_csv_export_insufficient_data() {
        let candles = generate_test_candles(10);
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.csv");

        let exporter = CsvExporter::new();
        let config = ExportConfig::default()
            .with_timeframes(vec![Timeframe::Min1])
            .with_min_candles(Timeframe::Min1, 100);

        let result = exporter.export(&candles, &path, &config);
        assert!(matches!(result, Err(ExportError::InsufficientData(_))));
    }

    #[test]
    fn test_csv_export_basic() {
        // Generate enough candles for meaningful analysis
        let candles = generate_test_candles(1000);
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.csv");

        let exporter = CsvExporter::new();
        let config = ExportConfig::default()
            .with_timeframes(vec![Timeframe::Min1, Timeframe::Min5])
            .with_min_candles(Timeframe::Min1, 50)
            .with_min_candles(Timeframe::Min5, 10)
            .with_min_active_levels(0) // Don't require active levels for test
            .with_label_lookahead(5);

        let result = exporter.export(&candles, &path, &config);
        assert!(result.is_ok(), "Export failed: {:?}", result);

        let rows = result.unwrap();
        assert!(rows > 0, "Expected some rows to be written");

        // Verify file exists and has content
        let content = fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines.len() > 1, "Expected header + data rows");
        // Default config uses numbered features
        assert!(lines[0].starts_with("f0,"), "Header should start with f0 (numbered features)");
        assert!(lines[0].contains("direction_up"), "Header should contain direction_up label");
    }
}

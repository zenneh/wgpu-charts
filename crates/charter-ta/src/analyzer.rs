//! Main analyzer that orchestrates ranges, levels, and rules.

use charter_core::Candle;

use crate::rules::RuleContext;
use crate::types::{
    CandleMetadata, Level, LevelEvent, LevelId, LevelTracker, Range, RangeBuilder,
};

/// Configuration for the analyzer.
#[derive(Debug, Clone)]
pub struct AnalyzerConfig {
    /// Threshold for doji detection (body_ratio < threshold = doji).
    pub doji_threshold: f32,
    /// Minimum candles required to form a valid range.
    pub min_range_candles: usize,
    /// Tolerance for level interactions (in price units).
    pub level_tolerance: f32,
    /// Whether to create greedy hold levels.
    pub create_greedy_levels: bool,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            doji_threshold: 0.001,
            min_range_candles: 2,
            level_tolerance: 0.0,
            create_greedy_levels: true,
        }
    }
}

impl AnalyzerConfig {
    /// Create a new config with custom settings.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn doji_threshold(mut self, threshold: f32) -> Self {
        self.doji_threshold = threshold;
        self
    }

    pub fn min_range_candles(mut self, min: usize) -> Self {
        self.min_range_candles = min;
        self
    }

    pub fn level_tolerance(mut self, tolerance: f32) -> Self {
        self.level_tolerance = tolerance;
        self
    }

    pub fn create_greedy_levels(mut self, create: bool) -> Self {
        self.create_greedy_levels = create;
        self
    }
}

/// Result of processing a single candle.
#[derive(Debug, Default)]
pub struct AnalysisResult {
    /// Ranges that were completed on this candle.
    pub completed_ranges: Vec<Range>,
    /// Levels that were created on this candle.
    pub created_levels: Vec<LevelId>,
    /// Level events (hits, breaks) that occurred.
    pub level_events: Vec<LevelEvent>,
}

impl AnalysisResult {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if any ranges were completed.
    pub fn has_new_ranges(&self) -> bool {
        !self.completed_ranges.is_empty()
    }

    /// Check if any level events occurred.
    pub fn has_level_events(&self) -> bool {
        !self.level_events.is_empty()
    }
}

/// Main analyzer for technical analysis.
///
/// Processes candles incrementally and tracks ranges, levels, and their interactions.
pub struct Analyzer {
    config: AnalyzerConfig,

    /// All candles processed so far.
    candles: Vec<Candle>,
    /// Pre-computed metadata for each candle.
    metadata: Vec<CandleMetadata>,

    /// Builder for detecting ranges.
    range_builder: RangeBuilder,
    /// All completed ranges.
    ranges: Vec<Range>,

    /// Tracker for active levels.
    level_tracker: LevelTracker,
}

impl Analyzer {
    /// Create a new analyzer with default configuration.
    pub fn new() -> Self {
        Self::with_config(AnalyzerConfig::default())
    }

    /// Create a new analyzer with custom configuration.
    pub fn with_config(config: AnalyzerConfig) -> Self {
        let level_tracker = LevelTracker::new(config.level_tolerance, config.create_greedy_levels);
        let range_builder = RangeBuilder::new(config.doji_threshold);

        Self {
            config,
            candles: Vec::new(),
            metadata: Vec::new(),
            range_builder,
            ranges: Vec::new(),
            level_tracker,
        }
    }

    /// Create a builder for configuring the analyzer.
    pub fn builder() -> AnalyzerBuilder {
        AnalyzerBuilder::new()
    }

    /// Process a single candle.
    ///
    /// Returns the analysis result for this candle.
    pub fn process_candle(&mut self, candle: Candle) -> AnalysisResult {
        let index = self.candles.len();
        let mut result = AnalysisResult::new();

        // Compute and store metadata
        let meta = CandleMetadata::from_candle(&candle, self.config.doji_threshold);
        self.metadata.push(meta);
        self.candles.push(candle);

        // Process range detection
        if let Some(completed_range) = self.range_builder.process(index, &self.candles[index]) {
            // Check if range meets minimum candle requirement
            if completed_range.candle_count >= self.config.min_range_candles {
                // Create levels from the completed range
                let level_count_before = self.level_tracker.levels.len();
                self.level_tracker.create_levels_from_range(&completed_range, index);

                // Record created levels
                for level in self.level_tracker.levels.iter().skip(level_count_before) {
                    result.created_levels.push(level.id);
                    result.level_events.push(LevelEvent::Created { level_id: level.id });
                }

                result.completed_ranges.push(completed_range.clone());
                self.ranges.push(completed_range);
            }
        }

        // Check level interactions
        let level_events = self.level_tracker.check_interactions(index, &self.candles[index]);
        result.level_events.extend(level_events);

        result
    }

    /// Process multiple candles in batch.
    pub fn process_batch(&mut self, candles: &[Candle]) -> Vec<AnalysisResult> {
        candles.iter().map(|c| self.process_candle(*c)).collect()
    }

    /// Get a rule context for the current state.
    ///
    /// Useful for evaluating custom rules.
    pub fn rule_context(&self) -> RuleContext<'_> {
        RuleContext::new(
            &self.candles,
            &self.metadata,
            self.candles.len().saturating_sub(1),
            &self.ranges,
            &self.level_tracker.levels,
        )
    }

    /// Get a rule context for a specific candle index.
    pub fn rule_context_at(&self, index: usize) -> Option<RuleContext<'_>> {
        if index >= self.candles.len() {
            return None;
        }
        Some(RuleContext::new(
            &self.candles,
            &self.metadata,
            index,
            &self.ranges,
            &self.level_tracker.levels,
        ))
    }

    /// Get all completed ranges.
    pub fn ranges(&self) -> &[Range] {
        &self.ranges
    }

    /// Get all active (non-broken) levels.
    pub fn active_levels(&self) -> impl Iterator<Item = &Level> {
        self.level_tracker.active_levels()
    }

    /// Get all levels (including broken ones).
    pub fn all_levels(&self) -> &[Level] {
        &self.level_tracker.levels
    }

    /// Get all processed candles.
    pub fn candles(&self) -> &[Candle] {
        &self.candles
    }

    /// Get all candle metadata.
    pub fn metadata(&self) -> &[CandleMetadata] {
        &self.metadata
    }

    /// Get a specific level by ID.
    pub fn level(&self, id: LevelId) -> Option<&Level> {
        self.level_tracker.levels.iter().find(|l| l.id == id)
    }

    /// Remove broken levels from tracking.
    pub fn prune_broken_levels(&mut self) {
        self.level_tracker.prune_broken();
    }

    /// Reset the analyzer to initial state.
    pub fn reset(&mut self) {
        self.candles.clear();
        self.metadata.clear();
        self.range_builder.reset();
        self.ranges.clear();
        self.level_tracker.clear();
    }

    /// Get the current configuration.
    pub fn config(&self) -> &AnalyzerConfig {
        &self.config
    }

    /// Number of candles processed.
    pub fn candle_count(&self) -> usize {
        self.candles.len()
    }

    /// Number of completed ranges.
    pub fn range_count(&self) -> usize {
        self.ranges.len()
    }

    /// Number of active levels.
    pub fn active_level_count(&self) -> usize {
        self.level_tracker.levels.iter().filter(|l| l.is_active()).count()
    }
}

impl Default for Analyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating an Analyzer with custom configuration.
pub struct AnalyzerBuilder {
    config: AnalyzerConfig,
}

impl AnalyzerBuilder {
    pub fn new() -> Self {
        Self {
            config: AnalyzerConfig::default(),
        }
    }

    pub fn doji_threshold(mut self, threshold: f32) -> Self {
        self.config.doji_threshold = threshold;
        self
    }

    pub fn min_range_candles(mut self, min: usize) -> Self {
        self.config.min_range_candles = min;
        self
    }

    pub fn level_tolerance(mut self, tolerance: f32) -> Self {
        self.config.level_tolerance = tolerance;
        self
    }

    pub fn create_greedy_levels(mut self, create: bool) -> Self {
        self.config.create_greedy_levels = create;
        self
    }

    pub fn build(self) -> Analyzer {
        Analyzer::with_config(self.config)
    }
}

impl Default for AnalyzerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CandleDirection;

    fn make_candle(open: f32, high: f32, low: f32, close: f32) -> Candle {
        Candle::new(0.0, open, high, low, close, 100.0)
    }

    #[test]
    fn test_analyzer_basic() {
        let mut analyzer = Analyzer::builder()
            .min_range_candles(2)
            .create_greedy_levels(false)
            .build();

        // Add 3 bullish candles
        analyzer.process_candle(make_candle(100.0, 110.0, 95.0, 105.0));
        analyzer.process_candle(make_candle(105.0, 115.0, 100.0, 110.0));
        analyzer.process_candle(make_candle(110.0, 120.0, 105.0, 115.0));

        // No range yet (still building)
        assert_eq!(analyzer.range_count(), 0);

        // Add a bearish candle to complete the bullish range
        let result = analyzer.process_candle(make_candle(115.0, 118.0, 108.0, 110.0));

        // Should have completed the bullish range
        assert_eq!(result.completed_ranges.len(), 1);
        assert_eq!(result.completed_ranges[0].direction, CandleDirection::Bullish);
        assert_eq!(result.completed_ranges[0].candle_count, 3);

        // Level should be created
        assert!(analyzer.active_level_count() > 0);
    }

    #[test]
    fn test_analyzer_level_hit() {
        let mut analyzer = Analyzer::builder()
            .min_range_candles(2)
            .level_tolerance(0.5)
            .create_greedy_levels(false)
            .build();

        // Create a bearish range
        analyzer.process_candle(make_candle(110.0, 115.0, 100.0, 105.0)); // Bearish
        analyzer.process_candle(make_candle(105.0, 110.0, 95.0, 100.0));  // Bearish

        // Now a bullish candle to complete the bearish range
        let result = analyzer.process_candle(make_candle(100.0, 105.0, 98.0, 103.0));
        assert_eq!(result.completed_ranges.len(), 1);

        // The bearish range should create a level at min(first_low, last_low) = min(100, 95) = 95
        let levels: Vec<_> = analyzer.active_levels().collect();
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0].price, 95.0);
    }
}

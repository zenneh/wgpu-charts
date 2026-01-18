//! Dynamic indicator registry for managing technical indicators at runtime.
//!
//! This module provides a trait-object based approach for storing and managing
//! different indicator types polymorphically. It enables runtime addition and
//! removal of indicators without compile-time knowledge of specific types.

// Allow dead code for API surface - these methods will be used as more indicators are added
#![allow(dead_code)]

use charter_core::Candle;
use charter_indicators::{Indicator, IndicatorOutput, Macd, MacdConfig, MacdOutput};

/// A trait object wrapper for dynamic indicator dispatch.
///
/// This allows different indicator types to be stored and used polymorphically.
/// Implementors must be `Send + Sync` to support multi-threaded access patterns.
pub trait DynIndicator: Send + Sync {
    /// Calculate indicator values for the given candles.
    fn calculate(&self, candles: &[Candle]) -> IndicatorOutput;

    /// Get the human-readable name of this indicator.
    fn name(&self) -> &str;

    /// Whether this indicator should be overlaid on the price chart.
    ///
    /// Returns `true` for overlay indicators (e.g., moving averages),
    /// `false` for oscillators displayed in separate panels (e.g., MACD).
    fn is_overlay(&self) -> bool;

    /// Minimum number of periods required before the indicator produces valid output.
    fn min_periods(&self) -> usize;

    /// Get the indicator configuration as a type-erased reference.
    ///
    /// This allows UI code to inspect and potentially modify configuration.
    fn config_any(&self) -> &dyn std::any::Any;

    /// Get the indicator configuration as a mutable type-erased reference.
    fn config_any_mut(&mut self) -> &mut dyn std::any::Any;
}

/// GPU buffers for rendering MACD indicators.
///
/// MACD requires multiple GPU resources for its three components:
/// - MACD line (fast EMA - slow EMA)
/// - Signal line (EMA of MACD)
/// - Histogram (MACD - Signal)
pub struct MacdGpuBuffers {
    /// Buffer containing MACD line points.
    pub macd_line_buffer: wgpu::Buffer,
    /// Buffer containing signal line points.
    pub signal_line_buffer: wgpu::Buffer,
    /// Buffer containing histogram bar points.
    pub histogram_buffer: wgpu::Buffer,
    /// Parameters buffer for MACD line rendering.
    pub params_buffer: wgpu::Buffer,
    /// Parameters buffer for signal line rendering (different start index).
    pub signal_params_buffer: wgpu::Buffer,
    /// Bind group for MACD line rendering.
    pub macd_bind_group: wgpu::BindGroup,
    /// Bind group for signal line rendering.
    pub signal_bind_group: wgpu::BindGroup,
    /// Bind group for histogram rendering.
    pub histogram_bind_group: wgpu::BindGroup,
    /// Number of points in the MACD line buffer.
    pub macd_point_count: u32,
    /// Number of points in the signal line buffer.
    pub signal_point_count: u32,
    /// Number of points in the histogram buffer.
    pub histogram_point_count: u32,
    /// Candle index where MACD line data starts.
    pub macd_start_index: usize,
    /// Candle index where signal line data starts.
    pub signal_start_index: usize,
}

/// GPU buffers for an indicator's visualization.
///
/// This enum allows different indicator types to have their own
/// specialized GPU buffer layouts while being stored uniformly.
pub enum IndicatorGpuBuffers {
    /// GPU buffers for MACD indicator.
    Macd(MacdGpuBuffers),
    // Future indicator types can add their own variants here.
    // Example: Rsi(RsiGpuBuffers), BollingerBands(BbGpuBuffers), etc.
}

impl IndicatorGpuBuffers {
    /// Get the MACD-specific buffers if this is a MACD indicator.
    pub fn as_macd(&self) -> Option<&MacdGpuBuffers> {
        match self {
            IndicatorGpuBuffers::Macd(buffers) => Some(buffers),
        }
    }

    /// Get mutable MACD-specific buffers if this is a MACD indicator.
    pub fn as_macd_mut(&mut self) -> Option<&mut MacdGpuBuffers> {
        match self {
            IndicatorGpuBuffers::Macd(buffers) => Some(buffers),
        }
    }
}

/// A single instance of an indicator with its outputs and GPU resources.
///
/// This struct combines the indicator logic, computed outputs, and GPU
/// resources needed for rendering into a single manageable unit.
pub struct IndicatorInstance {
    /// Unique identifier for this instance.
    pub id: usize,
    /// Human-readable name/label for this instance.
    pub name: String,
    /// The indicator implementation (trait object).
    pub indicator: Box<dyn DynIndicator>,
    /// Computed outputs per timeframe. Index corresponds to timeframe index.
    pub outputs: Vec<Option<IndicatorOutput>>,
    /// MACD-specific outputs per timeframe (for direct access to typed data).
    /// This is needed because `IndicatorOutput::MultiLine` loses type info.
    pub macd_outputs: Vec<Option<MacdOutput>>,
    /// GPU buffers for rendering.
    pub gpu_buffers: Option<IndicatorGpuBuffers>,
}

impl IndicatorInstance {
    /// Create a new indicator instance.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this instance
    /// * `name` - Human-readable name/label
    /// * `indicator` - The indicator implementation
    /// * `num_timeframes` - Number of timeframes to pre-allocate output slots for
    pub fn new(
        id: usize,
        name: String,
        indicator: Box<dyn DynIndicator>,
        num_timeframes: usize,
    ) -> Self {
        Self {
            id,
            name,
            indicator,
            outputs: vec![None; num_timeframes],
            macd_outputs: vec![None; num_timeframes],
            gpu_buffers: None,
        }
    }

    /// Calculate the indicator for the given candles and store in outputs.
    pub fn compute(&mut self, candles: &[Candle], timeframe: usize) {
        if candles.is_empty() {
            self.outputs[timeframe] = None;
            self.macd_outputs[timeframe] = None;
            return;
        }

        let output = self.indicator.calculate(candles);
        self.outputs[timeframe] = Some(output);
    }

    /// Check if this indicator is enabled.
    ///
    /// For MACD indicators, this checks the config's enabled field.
    pub fn is_enabled(&self) -> bool {
        if let Some(config) = self.indicator.config_any().downcast_ref::<MacdConfig>() {
            config.enabled
        } else {
            // Default to enabled for unknown indicator types
            true
        }
    }

    /// Get MACD configuration if this is a MACD indicator.
    pub fn macd_config(&self) -> Option<&MacdConfig> {
        self.indicator.config_any().downcast_ref::<MacdConfig>()
    }

    /// Get mutable MACD configuration if this is a MACD indicator.
    pub fn macd_config_mut(&mut self) -> Option<&mut MacdConfig> {
        self.indicator.config_any_mut().downcast_mut::<MacdConfig>()
    }
}

/// Registry for managing indicator instances.
///
/// The registry owns all indicator instances and provides methods for
/// adding, removing, and iterating over them. It handles ID generation
/// and maintains the relationship between instances and their GPU resources.
pub struct IndicatorRegistry {
    /// All registered indicator instances.
    instances: Vec<IndicatorInstance>,
    /// Next available unique ID for new instances.
    next_id: usize,
}

impl Default for IndicatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl IndicatorRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            next_id: 0,
        }
    }

    /// Add a new indicator to the registry.
    ///
    /// Returns the unique ID assigned to the new instance.
    ///
    /// # Arguments
    ///
    /// * `indicator` - The indicator implementation
    /// * `name` - Human-readable name for this instance
    /// * `num_timeframes` - Number of timeframes for output pre-allocation
    pub fn add<I: DynIndicator + 'static>(
        &mut self,
        indicator: I,
        name: String,
        num_timeframes: usize,
    ) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        let instance = IndicatorInstance::new(id, name, Box::new(indicator), num_timeframes);
        self.instances.push(instance);

        id
    }

    /// Remove an indicator by its index in the registry.
    ///
    /// Note: This uses index-based removal. The caller must ensure
    /// the index is valid and update any cached indices after removal.
    pub fn remove(&mut self, index: usize) {
        if index < self.instances.len() {
            self.instances.remove(index);
        }
    }

    /// Remove an indicator by its unique ID.
    ///
    /// Returns the removed instance if found, or None if not found.
    pub fn remove_by_id(&mut self, id: usize) -> Option<IndicatorInstance> {
        if let Some(pos) = self.instances.iter().position(|i| i.id == id) {
            Some(self.instances.remove(pos))
        } else {
            None
        }
    }

    /// Get an indicator instance by its index.
    pub fn get(&self, index: usize) -> Option<&IndicatorInstance> {
        self.instances.get(index)
    }

    /// Get a mutable indicator instance by its index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut IndicatorInstance> {
        self.instances.get_mut(index)
    }

    /// Get an indicator instance by its unique ID.
    pub fn get_by_id(&self, id: usize) -> Option<&IndicatorInstance> {
        self.instances.iter().find(|i| i.id == id)
    }

    /// Get a mutable indicator instance by its unique ID.
    pub fn get_by_id_mut(&mut self, id: usize) -> Option<&mut IndicatorInstance> {
        self.instances.iter_mut().find(|i| i.id == id)
    }

    /// Iterate over all indicator instances.
    pub fn iter(&self) -> impl Iterator<Item = &IndicatorInstance> {
        self.instances.iter()
    }

    /// Iterate mutably over all indicator instances.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut IndicatorInstance> {
        self.instances.iter_mut()
    }

    /// Get the number of registered indicators.
    #[must_use]
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Check if the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Recompute all indicators for a specific timeframe.
    ///
    /// This is called when switching timeframes or when new data arrives.
    pub fn recompute_all(&mut self, candles: &[Candle], timeframe: usize) {
        for instance in &mut self.instances {
            instance.compute(candles, timeframe);
        }
    }

    /// Get the index of an instance by its ID.
    pub fn index_of(&self, id: usize) -> Option<usize> {
        self.instances.iter().position(|i| i.id == id)
    }
}

// =============================================================================
// DynIndicator implementations for specific indicator types
// =============================================================================

/// Wrapper struct for MACD that holds configuration.
///
/// This is needed because we want to store and modify the config,
/// but the underlying `Macd` struct from charter-indicators only
/// holds an immutable reference to config.
pub struct DynMacd {
    config: MacdConfig,
}

impl DynMacd {
    /// Create a new dynamic MACD indicator with the given configuration.
    #[must_use]
    pub fn new(config: MacdConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &MacdConfig {
        &self.config
    }

    /// Get mutable configuration.
    pub fn config_mut(&mut self) -> &mut MacdConfig {
        &mut self.config
    }

    /// Generate a label string for this MACD configuration.
    #[must_use]
    pub fn label(&self) -> String {
        format!(
            "MACD({},{},{})",
            self.config.fast_period, self.config.slow_period, self.config.signal_period
        )
    }

    /// Calculate MACD and return the structured output.
    ///
    /// This returns `MacdOutput` directly for cases where the typed
    /// output is needed (e.g., for GPU buffer creation).
    pub fn calculate_macd(&self, candles: &[Candle]) -> MacdOutput {
        let macd = Macd::new(self.config.clone());
        macd.calculate_macd(candles)
    }
}

impl DynIndicator for DynMacd {
    fn calculate(&self, candles: &[Candle]) -> IndicatorOutput {
        let macd = Macd::new(self.config.clone());
        Indicator::calculate(&macd, candles)
    }

    fn name(&self) -> &str {
        "MACD"
    }

    fn is_overlay(&self) -> bool {
        false // MACD is shown in a separate panel, not on the price chart
    }

    fn min_periods(&self) -> usize {
        // Need slow_period for first MACD value, then signal_period more for signal line
        self.config.slow_period + self.config.signal_period - 1
    }

    fn config_any(&self) -> &dyn std::any::Any {
        &self.config
    }

    fn config_any_mut(&mut self) -> &mut dyn std::any::Any {
        &mut self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles(count: usize) -> Vec<Candle> {
        (0..count)
            .map(|i| Candle {
                timestamp: i as f64,
                open: 100.0 + i as f32,
                high: 101.0 + i as f32,
                low: 99.0 + i as f32,
                close: 100.0 + i as f32,
                volume: 1000.0,
            })
            .collect()
    }

    #[test]
    fn test_registry_add_and_remove() {
        let mut registry = IndicatorRegistry::new();
        assert!(registry.is_empty());

        let id1 = registry.add(DynMacd::new(MacdConfig::default()), "MACD 1".to_string(), 1);
        assert_eq!(registry.len(), 1);
        assert_eq!(id1, 0);

        let id2 = registry.add(DynMacd::new(MacdConfig::default()), "MACD 2".to_string(), 1);
        assert_eq!(registry.len(), 2);
        assert_eq!(id2, 1);

        registry.remove(0);
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get(0).unwrap().id, 1); // The second instance is now at index 0
    }

    #[test]
    fn test_registry_get_by_id() {
        let mut registry = IndicatorRegistry::new();

        let id = registry.add(DynMacd::new(MacdConfig::default()), "Test MACD".to_string(), 1);

        let instance = registry.get_by_id(id).unwrap();
        assert_eq!(instance.name, "Test MACD");
        assert_eq!(instance.id, id);

        assert!(registry.get_by_id(999).is_none());
    }

    #[test]
    fn test_dyn_macd_calculate() {
        let macd = DynMacd::new(MacdConfig::default());
        let candles = make_candles(50);

        let output = macd.calculate(&candles);
        assert!(matches!(output, IndicatorOutput::MultiLine(_)));

        assert_eq!(macd.name(), "MACD");
        assert!(!macd.is_overlay());
        assert_eq!(macd.min_periods(), 34); // 26 + 9 - 1
    }

    #[test]
    fn test_instance_compute() {
        let mut registry = IndicatorRegistry::new();
        registry.add(DynMacd::new(MacdConfig::default()), "MACD".to_string(), 2);

        let candles = make_candles(50);
        registry.recompute_all(&candles, 0);

        let instance = registry.get(0).unwrap();
        assert!(instance.outputs[0].is_some());
        assert!(instance.outputs[1].is_none()); // Not computed yet
    }

    #[test]
    fn test_macd_config_access() {
        let mut registry = IndicatorRegistry::new();
        registry.add(
            DynMacd::new(MacdConfig {
                fast_period: 8,
                ..MacdConfig::default()
            }),
            "Custom MACD".to_string(),
            1,
        );

        let instance = registry.get(0).unwrap();
        let config = instance.macd_config().unwrap();
        assert_eq!(config.fast_period, 8);

        // Test mutable access
        let instance = registry.get_mut(0).unwrap();
        if let Some(config) = instance.macd_config_mut() {
            config.fast_period = 10;
        }

        let instance = registry.get(0).unwrap();
        let config = instance.macd_config().unwrap();
        assert_eq!(config.fast_period, 10);
    }
}

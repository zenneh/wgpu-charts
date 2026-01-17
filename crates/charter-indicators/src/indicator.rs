//! Core indicator traits and types.

use charter_core::{Candle, TimeSeries};

/// Trait for indicator configuration.
pub trait IndicatorConfig: Clone + Default {}

/// Which price to use for indicator calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PriceSource {
    Open,
    High,
    Low,
    #[default]
    Close,
    /// (High + Low) / 2
    HL2,
    /// (High + Low + Close) / 3
    HLC3,
    /// (Open + High + Low + Close) / 4
    OHLC4,
}

impl PriceSource {
    /// Extract the price from a candle based on this source.
    pub fn extract(&self, candle: &Candle) -> f32 {
        match self {
            PriceSource::Open => candle.open,
            PriceSource::High => candle.high,
            PriceSource::Low => candle.low,
            PriceSource::Close => candle.close,
            PriceSource::HL2 => (candle.high + candle.low) / 2.0,
            PriceSource::HLC3 => (candle.high + candle.low + candle.close) / 3.0,
            PriceSource::OHLC4 => (candle.open + candle.high + candle.low + candle.close) / 4.0,
        }
    }
}

/// Output from an indicator calculation.
#[derive(Debug, Clone)]
pub enum IndicatorOutput {
    /// Single line output (e.g., SMA, EMA).
    Line(TimeSeries<f32>),
    /// Multiple named lines (e.g., Bollinger Bands, MACD).
    MultiLine(Vec<(String, TimeSeries<f32>)>),
    /// Oscillator with values and bounds (e.g., RSI).
    Oscillator {
        values: TimeSeries<f32>,
        upper_bound: f32,
        lower_bound: f32,
    },
}

/// Trait for technical indicators.
pub trait Indicator {
    /// The configuration type for this indicator.
    type Config: IndicatorConfig;

    /// Create a new indicator with the given configuration.
    fn new(config: Self::Config) -> Self;

    /// Calculate the indicator values for the given candles.
    fn calculate(&self, candles: &[Candle]) -> IndicatorOutput;

    /// Minimum number of periods required before the indicator produces valid output.
    fn min_periods(&self) -> usize;

    /// Whether this indicator should be overlaid on the price chart (true)
    /// or displayed in a separate pane (false).
    fn is_overlay(&self) -> bool;

    /// Human-readable name of the indicator.
    fn name(&self) -> &str;
}

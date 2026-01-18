//! MACD (Moving Average Convergence Divergence) indicator.

use charter_core::{Candle, TimeSeries};

use crate::indicator::{Indicator, IndicatorConfig, IndicatorOutput, PriceSource};

/// MACD indicator configuration.
#[derive(Debug, Clone)]
pub struct MacdConfig {
    /// Fast EMA period (default: 12).
    pub fast_period: usize,
    /// Slow EMA period (default: 26).
    pub slow_period: usize,
    /// Signal line EMA period (default: 9).
    pub signal_period: usize,
    /// Price source for calculation.
    pub price_source: PriceSource,
    /// Whether this MACD instance is enabled.
    pub enabled: bool,
    /// Color for MACD line (RGB).
    pub macd_color: [f32; 3],
    /// Color for signal line (RGB).
    pub signal_color: [f32; 3],
    /// Color for histogram positive (RGB).
    pub histogram_pos_color: [f32; 3],
    /// Color for histogram negative (RGB).
    pub histogram_neg_color: [f32; 3],
}

impl Default for MacdConfig {
    fn default() -> Self {
        Self {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
            price_source: PriceSource::Close,
            enabled: true,
            macd_color: [0.2, 0.6, 1.0],        // Blue
            signal_color: [1.0, 0.5, 0.2],      // Orange
            histogram_pos_color: [0.2, 0.8, 0.4], // Green
            histogram_neg_color: [0.8, 0.2, 0.2], // Red
        }
    }
}

impl IndicatorConfig for MacdConfig {}

/// MACD indicator output.
#[derive(Debug, Clone)]
pub struct MacdOutput {
    /// MACD line values (fast EMA - slow EMA).
    pub macd_line: TimeSeries<f32>,
    /// Signal line values (EMA of MACD line).
    pub signal_line: TimeSeries<f32>,
    /// Histogram values (MACD - Signal).
    pub histogram: TimeSeries<f32>,
}

/// MACD indicator.
pub struct Macd {
    config: MacdConfig,
}

impl Indicator for Macd {
    type Config = MacdConfig;

    fn new(config: Self::Config) -> Self {
        Self { config }
    }

    fn calculate(&self, candles: &[Candle]) -> IndicatorOutput {
        let output = self.calculate_macd(candles);

        // Convert to MultiLine format for rendering
        IndicatorOutput::MultiLine(vec![
            ("MACD".to_string(), output.macd_line),
            ("Signal".to_string(), output.signal_line),
            ("Histogram".to_string(), output.histogram),
        ])
    }

    fn min_periods(&self) -> usize {
        // Need slow_period for first MACD value, then signal_period more for signal line
        self.config.slow_period + self.config.signal_period - 1
    }

    fn is_overlay(&self) -> bool {
        // MACD is displayed in a separate pane, not overlaid on price
        false
    }

    fn name(&self) -> &str {
        "MACD"
    }
}

impl Macd {
    /// Calculate MACD values and return structured output.
    pub fn calculate_macd(&self, candles: &[Candle]) -> MacdOutput {
        let prices: Vec<f32> = candles
            .iter()
            .map(|c| self.config.price_source.extract(c))
            .collect();

        if prices.len() < self.config.slow_period {
            return MacdOutput {
                macd_line: TimeSeries::new(),
                signal_line: TimeSeries::new(),
                histogram: TimeSeries::new(),
            };
        }

        // Calculate fast and slow EMAs
        let fast_ema = calculate_ema(&prices, self.config.fast_period);
        let slow_ema = calculate_ema(&prices, self.config.slow_period);

        // MACD line = Fast EMA - Slow EMA
        // Start index is where slow EMA starts (slow_period - 1)
        let macd_start = self.config.slow_period - 1;
        let mut macd_values: Vec<Option<f32>> = Vec::with_capacity(prices.len() - macd_start);

        for i in macd_start..prices.len() {
            let fast_idx = i - (self.config.fast_period - 1);
            let slow_idx = i - macd_start;

            if fast_idx < fast_ema.len() && slow_idx < slow_ema.len() {
                macd_values.push(Some(fast_ema[fast_idx] - slow_ema[slow_idx]));
            } else {
                macd_values.push(None);
            }
        }

        // Signal line = EMA of MACD line
        let macd_for_signal: Vec<f32> = macd_values.iter().filter_map(|v| *v).collect();
        let signal_ema = calculate_ema(&macd_for_signal, self.config.signal_period);

        // Build signal line TimeSeries
        let signal_start = macd_start + self.config.signal_period - 1;
        let mut signal_values: Vec<Option<f32>> = Vec::with_capacity(prices.len() - signal_start);

        for i in 0..(prices.len() - signal_start) {
            if i < signal_ema.len() {
                signal_values.push(Some(signal_ema[i]));
            } else {
                signal_values.push(None);
            }
        }

        // Histogram = MACD - Signal
        let mut histogram_values: Vec<Option<f32>> = Vec::with_capacity(prices.len() - signal_start);

        for i in 0..(prices.len() - signal_start) {
            let macd_idx = i + self.config.signal_period - 1;
            if macd_idx < macd_values.len() {
                if let (Some(m), Some(s)) = (macd_values[macd_idx], signal_values.get(i).and_then(|v| *v)) {
                    histogram_values.push(Some(m - s));
                } else {
                    histogram_values.push(None);
                }
            } else {
                histogram_values.push(None);
            }
        }

        MacdOutput {
            macd_line: TimeSeries::with_offset(macd_values, macd_start),
            signal_line: TimeSeries::with_offset(signal_values, signal_start),
            histogram: TimeSeries::with_offset(histogram_values, signal_start),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &MacdConfig {
        &self.config
    }
}

/// Calculate Exponential Moving Average.
fn calculate_ema(prices: &[f32], period: usize) -> Vec<f32> {
    if prices.len() < period || period == 0 {
        return Vec::new();
    }

    let multiplier = 2.0 / (period as f32 + 1.0);
    let mut ema_values = Vec::with_capacity(prices.len() - period + 1);

    // First EMA value is SMA of first `period` prices
    let first_sma: f32 = prices[..period].iter().sum::<f32>() / period as f32;
    ema_values.push(first_sma);

    // Calculate subsequent EMA values
    for i in period..prices.len() {
        let prev_ema = ema_values.last().unwrap();
        let new_ema = (prices[i] - prev_ema) * multiplier + prev_ema;
        ema_values.push(new_ema);
    }

    ema_values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles(closes: &[f32]) -> Vec<Candle> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &close)| Candle {
                timestamp: i as f64,
                open: close,
                high: close + 1.0,
                low: close - 1.0,
                close,
                volume: 100.0,
            })
            .collect()
    }

    #[test]
    fn test_ema_calculation() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ema = calculate_ema(&prices, 3);

        assert_eq!(ema.len(), 8); // 10 - 3 + 1 = 8
        assert!((ema[0] - 2.0).abs() < 0.001); // First value is SMA of [1,2,3] = 2.0
    }

    #[test]
    fn test_macd_basic() {
        // Create enough candles for MACD calculation
        let closes: Vec<f32> = (1..=50).map(|i| 100.0 + i as f32).collect();
        let candles = make_candles(&closes);

        let config = MacdConfig {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
            ..Default::default()
        };

        let macd = Macd::new(config);
        let output = macd.calculate_macd(&candles);

        // MACD line should start at index 25 (slow_period - 1)
        assert_eq!(output.macd_line.start_index(), 25);

        // Signal line should start at index 33 (slow_period + signal_period - 2)
        assert_eq!(output.signal_line.start_index(), 33);

        // Histogram should start at same index as signal
        assert_eq!(output.histogram.start_index(), 33);

        // Values should exist
        assert!(output.macd_line.get(25).is_some());
        assert!(output.signal_line.get(33).is_some());
        assert!(output.histogram.get(33).is_some());
    }

    #[test]
    fn test_macd_min_periods() {
        let config = MacdConfig {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
            ..Default::default()
        };

        let macd = Macd::new(config);
        assert_eq!(macd.min_periods(), 34); // 26 + 9 - 1
    }
}

//! Indicator framework for technical analysis.

pub mod indicator;
pub mod macd;

pub use indicator::{Indicator, IndicatorConfig, IndicatorOutput, PriceSource};
pub use macd::{Macd, MacdConfig, MacdOutput};

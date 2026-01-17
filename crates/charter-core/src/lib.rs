//! Core types for the charter application.
//!
//! This crate provides fundamental data structures with no external dependencies:
//! - `Candle` - OHLCV candle data
//! - `Timeframe` - Time period enumeration and aggregation
//! - `TimeSeries` - Container for indicator output

pub mod candle;
pub mod series;
pub mod timeframe;

pub use candle::{Candle, OHLCV};
pub use series::TimeSeries;
pub use timeframe::{aggregate_candles, Timeframe};

//! Candle data structures for OHLCV data.

/// OHLCV Candle data structure (CPU side).
#[derive(Debug, Clone, Copy)]
pub struct Candle {
    pub timestamp: f64,
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f32,
}

impl Candle {
    pub fn new(timestamp: f64, open: f32, high: f32, low: f32, close: f32, volume: f32) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }
}

/// Trait for types that provide OHLCV data.
pub trait OHLCV {
    fn open(&self) -> f32;
    fn high(&self) -> f32;
    fn low(&self) -> f32;
    fn close(&self) -> f32;
    fn volume(&self) -> f32;
}

impl OHLCV for Candle {
    fn open(&self) -> f32 {
        self.open
    }

    fn high(&self) -> f32 {
        self.high
    }

    fn low(&self) -> f32 {
        self.low
    }

    fn close(&self) -> f32 {
        self.close
    }

    fn volume(&self) -> f32 {
        self.volume
    }
}

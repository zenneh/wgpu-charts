//! Trade batch accumulator for efficient DuckDB inserts.

use std::time::Instant;

use crate::live::TradeData;

/// Accumulates trades and flushes on capacity or time interval.
pub struct TradeBatchAccumulator {
    buffer: Vec<TradeData>,
    capacity: usize,
    flush_interval_ms: u64,
    last_flush: Instant,
}

impl TradeBatchAccumulator {
    /// Create a new accumulator with given capacity and flush interval.
    pub fn new(capacity: usize, flush_interval_ms: u64) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            flush_interval_ms,
            last_flush: Instant::now(),
        }
    }

    /// Add a batch of trades. Returns a flush batch if capacity or interval exceeded.
    pub fn add_trades(&mut self, trades: &[TradeData]) -> Option<Vec<TradeData>> {
        self.buffer.extend_from_slice(trades);

        if self.should_flush() {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Check if a flush should occur (capacity exceeded or interval elapsed).
    pub fn should_flush(&self) -> bool {
        self.buffer.len() >= self.capacity
            || (!self.buffer.is_empty()
                && self.last_flush.elapsed().as_millis() >= self.flush_interval_ms as u128)
    }

    /// Flush the buffer, returning all accumulated trades.
    pub fn flush(&mut self) -> Vec<TradeData> {
        self.last_flush = Instant::now();
        std::mem::take(&mut self.buffer)
    }

    /// Number of trades currently buffered.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

impl Default for TradeBatchAccumulator {
    fn default() -> Self {
        Self::new(500, 2000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(price: f32) -> TradeData {
        TradeData {
            price,
            quantity: 1.0,
            is_buy: true,
            timestamp: 1000,
        }
    }

    #[test]
    fn test_flush_on_capacity() {
        let mut acc = TradeBatchAccumulator::new(3, 60_000);
        assert!(acc.add_trades(&[make_trade(1.0), make_trade(2.0)]).is_none());
        let batch = acc.add_trades(&[make_trade(3.0)]);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 3);
        assert!(acc.is_empty());
    }

    #[test]
    fn test_manual_flush() {
        let mut acc = TradeBatchAccumulator::new(100, 60_000);
        acc.add_trades(&[make_trade(1.0)]);
        assert_eq!(acc.len(), 1);
        let batch = acc.flush();
        assert_eq!(batch.len(), 1);
        assert!(acc.is_empty());
    }
}

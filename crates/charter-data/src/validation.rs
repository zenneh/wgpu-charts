//! Validation utilities for live market data.

use charter_core::Candle;

use crate::live::TradeData;

/// Validate a trade has reasonable values.
pub fn validate_trade(trade: &TradeData) -> bool {
    trade.price > 0.0
        && trade.price.is_finite()
        && trade.quantity > 0.0
        && trade.quantity.is_finite()
}

/// Validate a depth snapshot has reasonable values.
///
/// Checks that prices and quantities are finite and positive,
/// and that the book is not crossed (best bid < best ask).
pub fn validate_depth(bids: &[(f32, f32)], asks: &[(f32, f32)]) -> bool {
    // At least one side must have data
    if bids.is_empty() && asks.is_empty() {
        return false;
    }

    // Check all values are finite and positive
    for &(price, qty) in bids.iter().chain(asks.iter()) {
        if !price.is_finite() || price <= 0.0 || !qty.is_finite() || qty < 0.0 {
            return false;
        }
    }

    // Check book is not crossed.
    // MEXC sends bids/asks in varying sort order, so find actual best bid/ask.
    if !bids.is_empty() && !asks.is_empty() {
        let best_bid = bids.iter().map(|&(p, _)| p).fold(f32::MIN, f32::max);
        let best_ask = asks.iter().map(|&(p, _)| p).fold(f32::MAX, f32::min);
        if best_bid > best_ask * 1.001 {
            // Allow tiny overlap (0.1%) due to f32 precision and rapid updates
            return false;
        }
    }

    true
}

/// Validate a candle has reasonable values.
pub fn validate_candle(candle: &Candle) -> bool {
    candle.open.is_finite()
        && candle.high.is_finite()
        && candle.low.is_finite()
        && candle.close.is_finite()
        && candle.volume.is_finite()
        && candle.high >= candle.low
        && candle.open > 0.0
        && candle.close > 0.0
        && candle.low > 0.0
        && candle.volume >= 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_trade_valid() {
        let trade = TradeData {
            price: 100.0,
            quantity: 1.5,
            is_buy: true,
            timestamp: 1000,
        };
        assert!(validate_trade(&trade));
    }

    #[test]
    fn test_validate_trade_zero_price() {
        let trade = TradeData {
            price: 0.0,
            quantity: 1.0,
            is_buy: true,
            timestamp: 1000,
        };
        assert!(!validate_trade(&trade));
    }

    #[test]
    fn test_validate_trade_nan() {
        let trade = TradeData {
            price: f32::NAN,
            quantity: 1.0,
            is_buy: true,
            timestamp: 1000,
        };
        assert!(!validate_trade(&trade));
    }

    #[test]
    fn test_validate_depth_valid() {
        let bids = vec![(99.0, 10.0), (98.0, 20.0)];
        let asks = vec![(100.0, 5.0), (101.0, 15.0)];
        assert!(validate_depth(&bids, &asks));
    }

    #[test]
    fn test_validate_depth_crossed_book() {
        // Significantly crossed (>0.1%) should fail
        let bids = vec![(101.0, 10.0)];
        let asks = vec![(100.0, 5.0)];
        assert!(!validate_depth(&bids, &asks));
    }

    #[test]
    fn test_validate_depth_tiny_overlap() {
        // Tiny overlap within 0.1% tolerance should pass (f32 precision / rapid updates)
        let bids = vec![(100.01, 10.0)];
        let asks = vec![(100.00, 5.0)];
        assert!(validate_depth(&bids, &asks));
    }

    #[test]
    fn test_validate_depth_empty() {
        assert!(!validate_depth(&[], &[]));
    }

    #[test]
    fn test_validate_candle_valid() {
        let candle = Candle::new(1000.0, 100.0, 105.0, 95.0, 102.0, 1000.0);
        assert!(validate_candle(&candle));
    }

    #[test]
    fn test_validate_candle_high_below_low() {
        let candle = Candle::new(1000.0, 100.0, 90.0, 95.0, 102.0, 1000.0);
        assert!(!validate_candle(&candle));
    }
}

//! MEXC API data source for loading candle data.

use charter_core::Candle;
use mexc_api::{types::KlineInterval, MexcClient, SpotApi};
use rust_decimal::prelude::ToPrimitive;

/// MEXC data source for fetching kline data.
pub struct MexcSource {
    symbol: String,
}

impl MexcSource {
    /// Create a new MEXC data source.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair symbol (e.g., "BTCUSDT")
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_uppercase(),
        }
    }

    /// Load initial candle data from MEXC API.
    ///
    /// Fetches recent 1-minute klines (last ~8 hours) to show data quickly.
    /// For more historical data, use `load_incremental` with a callback.
    pub async fn load(&self) -> anyhow::Result<Vec<Candle>> {
        let client = MexcClient::public()?;
        let spot = SpotApi::new(client);
        let market = spot.market();

        // First, test connectivity
        println!("Testing MEXC API connectivity...");
        match market.ping().await {
            Ok(_) => println!("  MEXC API ping successful"),
            Err(e) => {
                println!("  MEXC API ping failed: {}", e);
                return Err(anyhow::anyhow!("Failed to connect to MEXC API: {}", e));
            }
        }

        println!(
            "Fetching initial {} data from MEXC...",
            self.symbol
        );

        // Fetch most recent data first (no startTime = most recent)
        let klines = market
            .klines(
                &self.symbol,
                KlineInterval::OneMinute,
                None,
                None,
                Some(500), // Get initial batch quickly
            )
            .await?;

        let mut candles: Vec<Candle> = klines.iter().map(kline_to_candle).collect();

        // Sort by timestamp (oldest first)
        candles.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());

        println!(
            "Loaded {} initial candles from MEXC",
            candles.len()
        );

        Ok(candles)
    }
}

/// Convert a MEXC Kline to a Charter Candle.
fn kline_to_candle(kline: &mexc_api::types::Kline) -> Candle {
    Candle::new(
        (kline.open_time / 1000) as f64, // ms → seconds
        kline.open.0.to_f32().unwrap_or(0.0),
        kline.high.0.to_f32().unwrap_or(0.0),
        kline.low.0.to_f32().unwrap_or(0.0),
        kline.close.0.to_f32().unwrap_or(0.0),
        kline.volume.0.to_f32().unwrap_or(0.0),
    )
}

/// Convert a WebSocket kline to a Charter Candle.
pub fn ws_kline_to_candle(kline: &mexc_api::types::WsKline) -> Candle {
    Candle::new(
        (kline.time / 1000) as f64, // ms → seconds
        kline.open.0.to_f32().unwrap_or(0.0),
        kline.high.0.to_f32().unwrap_or(0.0),
        kline.low.0.to_f32().unwrap_or(0.0),
        kline.close.0.to_f32().unwrap_or(0.0),
        kline.volume.0.to_f32().unwrap_or(0.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mexc_source_creation() {
        let source = MexcSource::new("BTCUSDT");
        assert_eq!(source.symbol, "BTCUSDT");
    }

    #[test]
    fn test_mexc_source_uppercases_symbol() {
        let source = MexcSource::new("ethusdt");
        assert_eq!(source.symbol, "ETHUSDT");
    }
}

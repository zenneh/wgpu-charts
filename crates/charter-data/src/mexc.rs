//! MEXC API data source for loading candle and trade data.

use charter_core::Candle;
use mexc_api::{types::KlineInterval, MexcClient, SpotApi};
use rust_decimal::prelude::ToPrimitive;

use crate::live::TradeData;

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

    /// Fetch recent aggregate trades via REST API.
    ///
    /// Fetches trades from the last `minutes` minutes in batches of 1000.
    /// Returns a flat list of `TradeData` suitable for volume profile.
    pub async fn fetch_recent_trades(&self, minutes: u32) -> anyhow::Result<Vec<TradeData>> {
        let client = MexcClient::public()?;
        let spot = SpotApi::new(client);
        let market = spot.market();

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis() as i64;
        let start_ms = now_ms - (minutes as i64 * 60 * 1000);

        let mut all_trades = Vec::new();
        let mut current_start = start_ms;

        // Fetch in 1000-trade batches
        while current_start < now_ms {
            let batch = market
                .agg_trades(
                    &self.symbol,
                    None,
                    Some(current_start),
                    Some(now_ms),
                    Some(1000),
                )
                .await?;

            if batch.is_empty() {
                break;
            }

            let last_time = batch.last().map(|t| t.time).unwrap_or(now_ms);

            for t in &batch {
                let price = t.price.0.to_f32().unwrap_or(0.0);
                let quantity = t.qty.0.to_f32().unwrap_or(0.0);
                if price > 0.0 && quantity > 0.0 {
                    all_trades.push(TradeData {
                        price,
                        quantity,
                        is_buy: !t.is_buyer_maker,
                        timestamp: t.time,
                    });
                }
            }

            // Move window forward past the last trade we got
            if last_time <= current_start {
                break; // No progress, avoid infinite loop
            }
            current_start = last_time + 1;

            // Rate limit: small delay between batches
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        eprintln!(
            "[mexc] fetched {} historical trades for {} (last {} min)",
            all_trades.len(),
            self.symbol,
            minutes
        );

        Ok(all_trades)
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

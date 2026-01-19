//! Historical data fetcher for MEXC API.

use anyhow::{Context, Result};
use charter_core::Candle;
use mexc_api::{prelude::*, MexcClient, SpotApi};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

use crate::db::SyncTimeframe;

/// Batch of fetched candles with metadata.
#[derive(Debug)]
pub struct FetchBatch {
    /// The candles in this batch.
    pub candles: Vec<Candle>,
    /// Oldest timestamp in this batch (milliseconds).
    pub oldest_ts: i64,
    /// Whether this is the last batch (no more historical data available).
    pub is_last: bool,
}

/// Historical data fetcher.
pub struct HistoricalFetcher {
    pub(crate) api: SpotApi,
    symbol: String,
    batch_delay_ms: u64,
}

impl HistoricalFetcher {
    /// Create a new fetcher for the given symbol.
    pub fn new(symbol: &str, batch_delay_ms: u64) -> Result<Self> {
        let client = MexcClient::public().context("Failed to create MEXC client")?;
        let api = SpotApi::new(client);

        Ok(Self {
            api,
            symbol: symbol.to_string(),
            batch_delay_ms,
        })
    }

    /// Get current time in milliseconds.
    pub fn now_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
    }

    /// Convert SyncTimeframe to MEXC KlineInterval.
    fn to_kline_interval(tf: SyncTimeframe) -> KlineInterval {
        match tf {
            SyncTimeframe::Min1 => KlineInterval::OneMinute,
            SyncTimeframe::Min5 => KlineInterval::FiveMinutes,
            SyncTimeframe::Min15 => KlineInterval::FifteenMinutes,
            SyncTimeframe::Min30 => KlineInterval::ThirtyMinutes,
            SyncTimeframe::Hour1 => KlineInterval::OneHour,
            SyncTimeframe::Hour4 => KlineInterval::FourHours,
            SyncTimeframe::Day1 => KlineInterval::OneDay,
        }
    }

    /// Fetch a single batch of candles ending at the given timestamp for a specific timeframe.
    pub async fn fetch_batch_tf(
        &self,
        timeframe: SyncTimeframe,
        end_time_ms: Option<i64>,
    ) -> Result<FetchBatch> {
        let end_time = end_time_ms.unwrap_or_else(Self::now_ms);
        // Calculate start_time as 1000 candles before end_time
        let candle_duration_ms = timeframe.millis();
        let start_time = end_time - (1000 * candle_duration_ms);

        let interval = Self::to_kline_interval(timeframe);
        let klines = self
            .api
            .market()
            .klines(
                &self.symbol,
                interval,
                Some(start_time),
                Some(end_time),
                Some(1000), // Max batch size
            )
            .await
            .context("Failed to fetch klines")?;

        if klines.is_empty() {
            return Ok(FetchBatch {
                candles: vec![],
                oldest_ts: end_time,
                is_last: true,
            });
        }

        let candles: Vec<Candle> = klines
            .iter()
            .map(|k| Candle {
                timestamp: k.open_time as f64 / 1000.0, // Convert ms to seconds
                open: k.open.0.try_into().unwrap_or(0.0),
                high: k.high.0.try_into().unwrap_or(0.0),
                low: k.low.0.try_into().unwrap_or(0.0),
                close: k.close.0.try_into().unwrap_or(0.0),
                volume: k.volume.0.try_into().unwrap_or(0.0),
            })
            .collect();

        let oldest_ts = klines.iter().map(|k| k.open_time).min().unwrap_or(end_time);
        let is_last = klines.is_empty();

        log::debug!(
            "fetch_batch_tf({}): end_time={}, got {} candles, oldest_ts={}",
            timeframe.mexc_interval(),
            end_time,
            klines.len(),
            oldest_ts
        );

        Ok(FetchBatch {
            candles,
            oldest_ts,
            is_last,
        })
    }

    /// Fetch all historical data backwards from the given end time for a specific timeframe.
    pub async fn fetch_all_backwards_tf(
        &self,
        timeframe: SyncTimeframe,
        end_time_ms: Option<i64>,
        stop_at_ms: Option<i64>,
        tx: mpsc::Sender<Result<FetchBatch>>,
    ) {
        let mut current_end = end_time_ms.unwrap_or_else(Self::now_ms);
        let mut consecutive_empty = 0;
        let candle_duration_ms = timeframe.millis();

        log::info!(
            "Starting backward fetch ({}) from {} to {:?}",
            timeframe.mexc_interval(),
            current_end,
            stop_at_ms
        );

        loop {
            // Check if we should stop based on cutoff
            if let Some(stop_ts) = stop_at_ms {
                if current_end <= stop_ts {
                    log::info!(
                        "Reached cutoff timestamp for {}, stopping backward fetch",
                        timeframe.mexc_interval()
                    );
                    break;
                }
            }

            match self.fetch_batch_tf(timeframe, Some(current_end)).await {
                Ok(batch) => {
                    let oldest = batch.oldest_ts;
                    let candle_count = batch.candles.len();

                    // Track empty batches to avoid infinite loops
                    if candle_count == 0 {
                        consecutive_empty += 1;
                        if consecutive_empty >= 3 {
                            log::info!(
                                "Got 3 consecutive empty batches for {}, stopping",
                                timeframe.mexc_interval()
                            );
                            break;
                        }
                        // Try moving back in time
                        current_end -= 1000 * candle_duration_ms;
                        continue;
                    }
                    consecutive_empty = 0;

                    if tx.send(Ok(batch)).await.is_err() {
                        log::info!("Receiver dropped, stopping backward fetch");
                        break;
                    }

                    // Move to before the oldest candle in this batch
                    current_end = oldest - 1;

                    // Rate limit delay
                    if self.batch_delay_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(self.batch_delay_ms))
                            .await;
                    }
                }
                Err(e) => {
                    log::error!("Error in backward fetch: {}", e);
                    let _ = tx.send(Err(e)).await;
                    break;
                }
            }
        }
    }

    /// Fetch forward gap (from newest_db to now) for a specific timeframe.
    pub async fn fetch_forward_gap_tf(
        &self,
        timeframe: SyncTimeframe,
        from_ms: i64,
        tx: mpsc::Sender<Result<FetchBatch>>,
    ) {
        let now = Self::now_ms();
        let candle_duration_ms = timeframe.millis();
        let mut current_start = from_ms + candle_duration_ms;
        let interval = Self::to_kline_interval(timeframe);

        while current_start < now {
            let end_time = (current_start + 1000 * candle_duration_ms).min(now);

            match self
                .api
                .market()
                .klines(
                    &self.symbol,
                    interval,
                    Some(current_start),
                    Some(end_time),
                    Some(1000),
                )
                .await
            {
                Ok(klines) => {
                    if klines.is_empty() {
                        break;
                    }

                    let candles: Vec<Candle> = klines
                        .iter()
                        .map(|k| Candle {
                            timestamp: k.open_time as f64 / 1000.0,
                            open: k.open.0.try_into().unwrap_or(0.0),
                            high: k.high.0.try_into().unwrap_or(0.0),
                            low: k.low.0.try_into().unwrap_or(0.0),
                            close: k.close.0.try_into().unwrap_or(0.0),
                            volume: k.volume.0.try_into().unwrap_or(0.0),
                        })
                        .collect();

                    let newest_ts = klines.iter().map(|k| k.open_time).max().unwrap_or(end_time);
                    let is_last = newest_ts >= now - candle_duration_ms;

                    let batch = FetchBatch {
                        candles,
                        oldest_ts: current_start,
                        is_last,
                    };

                    if tx.send(Ok(batch)).await.is_err() {
                        break;
                    }

                    if is_last {
                        break;
                    }

                    current_start = newest_ts + candle_duration_ms;

                    if self.batch_delay_ms > 0 {
                        tokio::time::sleep(std::time::Duration::from_millis(self.batch_delay_ms))
                            .await;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(anyhow::anyhow!("{}", e))).await;
                    break;
                }
            }
        }
    }
}

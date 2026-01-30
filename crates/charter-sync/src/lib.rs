//! Historical data synchronization for charter.
//!
//! This crate provides functionality to sync historical candle data from MEXC API
//! to a local DuckDB database. It syncs multiple timeframes, prioritizing finer
//! resolution data and only using coarser timeframes for older historical data.

pub mod db;
pub mod fetcher;

use anyhow::Result;
use charter_core::Candle;
use db::SyncTimeframe;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

pub use db::{CandleDb, DepthLevel, TradeRecord};
pub use fetcher::{FetchBatch, HistoricalFetcher};

/// Sync state for the UI.
#[derive(Debug, Clone)]
pub enum SyncState {
    /// No sync in progress.
    Idle,
    /// Syncing in progress.
    Syncing {
        /// Number of candles fetched so far.
        fetched: u64,
        /// Estimated total candles (may be updated as we fetch).
        estimated_total: u64,
        /// Candles fetched per second.
        candles_per_sec: f32,
    },
    /// Sync completed successfully.
    Complete {
        /// Total candles in database.
        total_candles: u64,
    },
    /// Sync encountered an error.
    Error(String),
}

impl Default for SyncState {
    fn default() -> Self {
        Self::Idle
    }
}

/// Progress update sent during sync.
#[derive(Debug, Clone)]
pub struct SyncProgress {
    /// Number of candles fetched so far.
    pub fetched: u64,
    /// Estimated total candles.
    pub estimated_total: u64,
    /// Candles fetched per second.
    pub candles_per_sec: f32,
}

/// Sync manager for coordinating historical data sync.
pub struct SyncManager {
    db_path: PathBuf,
    batch_delay_ms: u64,
    sync_days: u32,
    cancel_flag: Arc<AtomicBool>,
}

impl SyncManager {
    /// Create a new sync manager.
    pub fn new(db_path: PathBuf, batch_delay_ms: u64, sync_days: u32) -> Self {
        Self {
            db_path,
            batch_delay_ms,
            sync_days,
            cancel_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get the default database path.
    pub fn default_db_path() -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("charter")
            .join("candles.duckdb")
    }

    /// Get the database path.
    pub fn db_path(&self) -> &PathBuf {
        &self.db_path
    }

    /// Open the database.
    pub fn open_db(&self) -> Result<CandleDb> {
        CandleDb::open(&self.db_path)
    }

    /// Check if sync should be cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::Relaxed)
    }

    /// Request cancellation of the current sync.
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
    }

    /// Reset the cancel flag for a new sync.
    pub fn reset_cancel(&self) {
        self.cancel_flag.store(false, Ordering::Relaxed);
    }

    /// Start syncing historical data for a symbol.
    /// Returns a receiver for progress updates.
    /// Requires a tokio runtime handle to spawn the async task.
    pub fn start_sync(
        &self,
        symbol: String,
        runtime: &tokio::runtime::Handle,
    ) -> (
        mpsc::Receiver<SyncProgress>,
        tokio::task::JoinHandle<Result<u64>>,
    ) {
        let (progress_tx, progress_rx) = mpsc::channel(100);
        let db_path = self.db_path.clone();
        let batch_delay = self.batch_delay_ms;
        let sync_days = self.sync_days;
        let cancel_flag = self.cancel_flag.clone();

        // Reset cancel flag
        cancel_flag.store(false, Ordering::Relaxed);

        let handle = runtime.spawn(async move {
            sync_symbol(symbol, db_path, batch_delay, sync_days, cancel_flag, progress_tx).await
        });

        (progress_rx, handle)
    }

    /// Load candles from the database for a symbol.
    pub fn load_candles(&self, symbol: &str) -> Result<Vec<Candle>> {
        let db = self.open_db()?;
        db.load_candles(symbol)
    }

    /// Get the candle count for a symbol.
    pub fn get_candle_count(&self, symbol: &str) -> Result<u64> {
        let db = self.open_db()?;
        db.get_candle_count(symbol)
    }
}

/// Perform the actual sync operation with multi-timeframe support.
/// Syncs progressively coarser timeframes for older data.
async fn sync_symbol(
    symbol: String,
    db_path: PathBuf,
    batch_delay_ms: u64,
    sync_days: u32,
    cancel_flag: Arc<AtomicBool>,
    progress_tx: mpsc::Sender<SyncProgress>,
) -> Result<u64> {
    // Calculate the cutoff timestamp (sync_days ago)
    let now_ms = fetcher::HistoricalFetcher::now_ms();
    let cutoff_ms = now_ms - (sync_days as i64 * 24 * 60 * 60 * 1000);

    // Open database - we'll reopen it for each batch to avoid Send issues
    let db = CandleDb::open(&db_path)?;
    let existing_count = db.get_candle_count(&symbol)?;
    drop(db); // Close the connection

    eprintln!(
        "[sync] starting {} sync: {} existing candles, {} days history",
        symbol, existing_count, sync_days
    );

    let start_time = Instant::now();
    let mut total_fetched: u64 = 0;

    // Estimate total based on sync_days (1 candle per minute as base)
    let estimated_total: u64 = (sync_days as u64) * 24 * 60;

    // Track where we've synced to for each timeframe.
    // Initialize from existing DB data rather than now_ms so we don't skip
    // ranges that a previous partial sync already covered.
    let mut oldest_synced_timestamp = {
        let db = CandleDb::open(&db_path)?;
        db.get_oldest_timestamp_any(&symbol)?
            .unwrap_or(now_ms)
    };

    // Timeframes to sync, from finest to coarsest
    let timeframes = SyncTimeframe::all();

    for &tf in timeframes {
        if cancel_flag.load(Ordering::Relaxed) {
            log::info!("Sync cancelled");
            break;
        }

        // Open DB to check existing data
        let db = CandleDb::open(&db_path)?;
        let oldest_tf = db.get_oldest_timestamp(&symbol, tf)?;
        let newest_tf = db.get_newest_timestamp(&symbol, tf)?;
        drop(db);

        log::info!(
            "Checking {} timeframe: oldest={:?}, newest={:?}, oldest_synced={}",
            tf.mexc_interval(),
            oldest_tf,
            newest_tf,
            oldest_synced_timestamp
        );
        eprintln!("[sync] {} timeframe: checking...", tf.mexc_interval());

        // Phase 1: Forward fill - get recent data up to now
        if let Some(newest) = newest_tf {
            if newest < now_ms - tf.millis() {
                log::info!(
                    "Forward filling {} from {} to now",
                    tf.mexc_interval(),
                    newest
                );
                let fetched = sync_timeframe_forward(
                    &db_path,
                    &symbol,
                    tf,
                    newest,
                    batch_delay_ms,
                    &cancel_flag,
                    &progress_tx,
                    &start_time,
                    total_fetched,
                    estimated_total,
                )
                .await?;
                total_fetched += fetched;
            }
        }

        // Phase 2: Backward fill - get historical data
        // Only sync data that's older than what we've already synced in finer timeframes
        let backward_start = oldest_tf.unwrap_or(oldest_synced_timestamp);

        // Only fetch backwards if we're still before the cutoff AND haven't already
        // synced this period with a finer timeframe
        if backward_start > cutoff_ms && backward_start <= oldest_synced_timestamp {
            log::info!(
                "Backward filling {} from {} to cutoff {}",
                tf.mexc_interval(),
                backward_start,
                cutoff_ms
            );

            let (fetched, new_oldest) = sync_timeframe_backward(
                &db_path,
                &symbol,
                tf,
                backward_start,
                cutoff_ms,
                batch_delay_ms,
                &cancel_flag,
                &progress_tx,
                &start_time,
                total_fetched,
                estimated_total,
            )
            .await?;

            total_fetched += fetched;

            // Update the oldest synced timestamp from what was actually fetched,
            // falling back to whatever is in the DB (covers partial fetches that
            // errored partway through).
            let effective_oldest = new_oldest.or_else(|| {
                CandleDb::open(&db_path)
                    .ok()
                    .and_then(|db| db.get_oldest_timestamp(&symbol, tf).ok().flatten())
            });
            if let Some(oldest) = effective_oldest {
                if oldest < oldest_synced_timestamp {
                    oldest_synced_timestamp = oldest;
                    log::info!(
                        "Updated oldest synced timestamp to {} after {} sync",
                        oldest_synced_timestamp,
                        tf.mexc_interval()
                    );
                }
            }
        }

        // Phase 3: Gap fill — find and fill internal gaps in existing data.
        // This catches gaps from previous partial syncs or API outages.
        {
            let db = CandleDb::open(&db_path)?;
            let gaps = db.find_gaps(&symbol, tf)?;
            drop(db);

            if !gaps.is_empty() {
                eprintln!(
                    "[sync] {} gap fill: {} gaps to fill",
                    tf.mexc_interval(),
                    gaps.len()
                );

                let gap_count = gaps.len();
                for (i, (gap_start, gap_end)) in gaps.iter().enumerate() {
                    if cancel_flag.load(Ordering::Relaxed) {
                        break;
                    }

                    let fetched = sync_gap_range(
                        &db_path,
                        &symbol,
                        tf,
                        *gap_start,
                        *gap_end,
                        batch_delay_ms,
                        &cancel_flag,
                    )
                    .await?;
                    total_fetched += fetched;

                    if fetched > 0 && (i + 1) % 10 == 0 {
                        eprintln!(
                            "[sync] {} gap fill: {}/{} gaps done ({} candles fetched so far)",
                            tf.mexc_interval(),
                            i + 1,
                            gap_count,
                            total_fetched
                        );
                    }
                }
                eprintln!(
                    "[sync] {} gap fill complete: {} total candles fetched",
                    tf.mexc_interval(),
                    total_fetched
                );
            }
        }

        // Phase 4: Validate + refetch loop.
        // Check OHLC consistency and close/open continuity. Invalid candles are
        // deleted, resulting gaps are re-fetched, then re-validated.
        const MAX_VALIDATION_ROUNDS: u32 = 3;
        for round in 0..MAX_VALIDATION_ROUNDS {
            if cancel_flag.load(Ordering::Relaxed) {
                break;
            }

            let db = CandleDb::open(&db_path)?;
            let (valid, deleted) = db.validate_candles(&symbol, tf, 0.001)?;
            if deleted > 0 {
                eprintln!(
                    "[sync] {} validation round {}: {} valid, {} deleted",
                    tf.mexc_interval(),
                    round + 1,
                    valid,
                    deleted
                );
            }
            log::info!(
                "{} validation round {}: {} valid, {} deleted",
                tf.mexc_interval(),
                round + 1,
                valid,
                deleted
            );

            if deleted == 0 {
                break;
            }

            // Find gaps left by deleted candles and re-fetch them
            let gaps = db.find_gaps(&symbol, tf)?;
            drop(db);

            if gaps.is_empty() {
                break;
            }

            eprintln!(
                "[sync] {} re-fetching {} gaps after validation",
                tf.mexc_interval(),
                gaps.len()
            );

            for (gap_start, gap_end) in &gaps {
                if cancel_flag.load(Ordering::Relaxed) {
                    break;
                }

                let fetched = sync_gap_range(
                    &db_path,
                    &symbol,
                    tf,
                    *gap_start,
                    *gap_end,
                    batch_delay_ms,
                    &cancel_flag,
                )
                .await?;
                total_fetched += fetched;
            }
        }

        // If we've reached the cutoff, no need to continue with coarser timeframes
        if oldest_synced_timestamp <= cutoff_ms {
            log::info!("Reached cutoff timestamp, stopping sync");
            break;
        }
    }

    // Final validation pass across all timeframes (no re-fetch, just clean up)
    {
        let db = CandleDb::open(&db_path)?;
        let mut total_valid: u64 = 0;
        let mut total_deleted: u64 = 0;
        for &tf in SyncTimeframe::all() {
            let (valid, deleted) = db.validate_candles(&symbol, tf, 0.001)?;
            total_valid += valid;
            total_deleted += deleted;
        }
        if total_valid > 0 || total_deleted > 0 {
            log::info!(
                "Final validation pass: {} valid, {} deleted",
                total_valid,
                total_deleted
            );
        }
    }

    let db = CandleDb::open(&db_path)?;
    let final_count = db.get_candle_count(&symbol)?;
    let elapsed = start_time.elapsed().as_secs_f32();
    eprintln!(
        "[sync] complete: {} candles total ({} new) in {:.1}s",
        final_count, total_fetched, elapsed
    );

    Ok(final_count)
}

/// Sync a timeframe forward (from newest to now).
async fn sync_timeframe_forward(
    db_path: &PathBuf,
    symbol: &str,
    timeframe: SyncTimeframe,
    from_ms: i64,
    batch_delay_ms: u64,
    cancel_flag: &Arc<AtomicBool>,
    progress_tx: &mpsc::Sender<SyncProgress>,
    start_time: &Instant,
    mut total_fetched: u64,
    estimated_total: u64,
) -> Result<u64> {
    let fetcher = HistoricalFetcher::new(symbol, batch_delay_ms)?;
    let (batch_tx, mut batch_rx) = mpsc::channel::<Result<FetchBatch>>(10);

    let cancel = cancel_flag.clone();

    tokio::spawn(async move {
        if !cancel.load(Ordering::Relaxed) {
            fetcher
                .fetch_forward_gap_tf(timeframe, from_ms, batch_tx)
                .await;
        }
    });

    let mut fetched_this_phase: u64 = 0;

    while let Some(result) = batch_rx.recv().await {
        if cancel_flag.load(Ordering::Relaxed) {
            break;
        }

        match result {
            Ok(batch) => {
                // Open DB, insert, close
                let db = CandleDb::open(db_path)?;
                let count = db.insert_candles(symbol, timeframe, &batch.candles)?;
                drop(db);

                fetched_this_phase += count as u64;
                total_fetched += count as u64;

                let elapsed = start_time.elapsed().as_secs_f32().max(0.1);
                let rate = total_fetched as f32 / elapsed;

                let _ = progress_tx
                    .send(SyncProgress {
                        fetched: total_fetched,
                        estimated_total,
                        candles_per_sec: rate,
                    })
                    .await;

                if batch.is_last {
                    break;
                }
            }
            Err(e) => {
                log::error!("Forward fetch batch failed: {:#}", e);
                break;
            }
        }
    }

    Ok(fetched_this_phase)
}

/// Sync a timeframe backward (from start to cutoff).
/// Returns (candles_fetched, oldest_timestamp_reached).
async fn sync_timeframe_backward(
    db_path: &PathBuf,
    symbol: &str,
    timeframe: SyncTimeframe,
    from_ms: i64,
    cutoff_ms: i64,
    batch_delay_ms: u64,
    cancel_flag: &Arc<AtomicBool>,
    progress_tx: &mpsc::Sender<SyncProgress>,
    start_time: &Instant,
    mut total_fetched: u64,
    estimated_total: u64,
) -> Result<(u64, Option<i64>)> {
    let fetcher = HistoricalFetcher::new(symbol, batch_delay_ms)?;
    let (batch_tx, mut batch_rx) = mpsc::channel::<Result<FetchBatch>>(10);

    let cancel = cancel_flag.clone();

    tokio::spawn(async move {
        if !cancel.load(Ordering::Relaxed) {
            fetcher
                .fetch_all_backwards_tf(timeframe, Some(from_ms - 1), Some(cutoff_ms), batch_tx)
                .await;
        }
    });

    let mut fetched_this_phase: u64 = 0;
    let mut oldest_reached: Option<i64> = None;

    while let Some(result) = batch_rx.recv().await {
        if cancel_flag.load(Ordering::Relaxed) {
            break;
        }

        match result {
            Ok(batch) => {
                if batch.candles.is_empty() {
                    break;
                }

                // Track the oldest timestamp we've received
                if oldest_reached.is_none() || batch.oldest_ts < oldest_reached.unwrap() {
                    oldest_reached = Some(batch.oldest_ts);
                }

                // Open DB, insert, close
                let db = CandleDb::open(db_path)?;
                let count = db.insert_candles(symbol, timeframe, &batch.candles)?;
                drop(db);

                fetched_this_phase += count as u64;
                total_fetched += count as u64;

                let elapsed = start_time.elapsed().as_secs_f32().max(0.1);
                let rate = total_fetched as f32 / elapsed;

                let _ = progress_tx
                    .send(SyncProgress {
                        fetched: total_fetched,
                        estimated_total,
                        candles_per_sec: rate,
                    })
                    .await;

                if batch.is_last {
                    break;
                }
            }
            Err(e) => {
                log::error!("Backward fetch batch failed: {:#}", e);
                break;
            }
        }
    }

    Ok((fetched_this_phase, oldest_reached))
}

/// Re-fetch a specific gap range for a timeframe.
/// Uses the klines API directly with explicit start/end times.
async fn sync_gap_range(
    db_path: &PathBuf,
    symbol: &str,
    timeframe: SyncTimeframe,
    gap_start_ms: i64,
    gap_end_ms: i64,
    batch_delay_ms: u64,
    cancel_flag: &Arc<AtomicBool>,
) -> Result<u64> {
    use mexc_api::{MexcClient, SpotApi};

    let client = MexcClient::public()
        .map_err(|e| anyhow::anyhow!("Failed to create MEXC client: {}", e))?;
    let api = SpotApi::new(client);
    let interval = HistoricalFetcher::kline_interval(timeframe);
    let candle_duration_ms = timeframe.millis();
    let mut current_start = gap_start_ms;
    let mut fetched_total: u64 = 0;
    let mut consecutive_errors = 0u32;
    const MAX_RETRIES: u32 = 3;

    while current_start <= gap_end_ms {
        if cancel_flag.load(Ordering::Relaxed) {
            break;
        }

        let end_time = (current_start + 1000 * candle_duration_ms).min(gap_end_ms + candle_duration_ms);

        match api
            .market()
            .klines(symbol, interval, Some(current_start), Some(end_time), Some(1000))
            .await
        {
            Ok(klines) => {
                consecutive_errors = 0;

                if klines.is_empty() {
                    log::debug!(
                        "Gap fill: empty response for [{}, {}]",
                        current_start,
                        end_time
                    );
                    // Skip past this range
                    current_start = end_time;
                    continue;
                }

                let candles: Vec<charter_core::Candle> = klines
                    .iter()
                    .map(|k| charter_core::Candle {
                        timestamp: k.open_time as f64 / 1000.0,
                        open: k.open.0.try_into().unwrap_or(0.0),
                        high: k.high.0.try_into().unwrap_or(0.0),
                        low: k.low.0.try_into().unwrap_or(0.0),
                        close: k.close.0.try_into().unwrap_or(0.0),
                        volume: k.volume.0.try_into().unwrap_or(0.0),
                    })
                    .collect();

                let newest_ts = klines.iter().map(|k| k.open_time).max().unwrap_or(end_time);

                let db = CandleDb::open(db_path)?;
                let count = db.insert_candles(symbol, timeframe, &candles)?;
                drop(db);

                fetched_total += count as u64;
                current_start = newest_ts + candle_duration_ms;

                if current_start > gap_end_ms {
                    break;
                }

                if batch_delay_ms > 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(batch_delay_ms)).await;
                }
            }
            Err(e) => {
                consecutive_errors += 1;
                if consecutive_errors >= MAX_RETRIES {
                    log::warn!(
                        "Gap fill [{} -> {}] failed after {} retries: {:#}",
                        gap_start_ms,
                        gap_end_ms,
                        MAX_RETRIES,
                        e
                    );
                    break;
                }
                let backoff_ms = 1000 * 2u64.pow(consecutive_errors - 1);
                log::warn!(
                    "Gap fill error (attempt {}/{}), retrying in {}ms: {:#}",
                    consecutive_errors,
                    MAX_RETRIES,
                    backoff_ms,
                    e
                );
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            }
        }
    }

    Ok(fetched_total)
}

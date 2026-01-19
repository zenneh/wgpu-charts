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

pub use db::CandleDb;
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

    log::info!(
        "Starting multi-timeframe sync for {}: existing={} candles, cutoff={}",
        symbol,
        existing_count,
        cutoff_ms
    );

    let start_time = Instant::now();
    let mut total_fetched: u64 = 0;

    // Estimate total based on sync_days (1 candle per minute as base)
    let estimated_total: u64 = (sync_days as u64) * 24 * 60;

    // Track where we've synced to for each timeframe
    // We start with the current time and work backwards
    let mut oldest_synced_timestamp = now_ms;

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

            // Update the oldest synced timestamp
            if let Some(oldest) = new_oldest {
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

        // If we've reached the cutoff, no need to continue with coarser timeframes
        if oldest_synced_timestamp <= cutoff_ms {
            log::info!("Reached cutoff timestamp, stopping sync");
            break;
        }
    }

    let db = CandleDb::open(&db_path)?;
    let final_count = db.get_candle_count(&symbol)?;
    log::info!(
        "Sync complete for {}: {} candles total ({} new)",
        symbol,
        final_count,
        total_fetched
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
                log::error!("Error in forward fetch: {}", e);
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
                log::error!("Error in backward fetch: {}", e);
                break;
            }
        }
    }

    Ok((fetched_this_phase, oldest_reached))
}

//! DuckDB storage for candle data.

use anyhow::{Context, Result};
use charter_core::Candle;
use duckdb::{params, Connection};
use std::collections::BTreeMap;
use std::path::Path;

/// Supported sync timeframes in minutes.
/// These map to MEXC API intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SyncTimeframe {
    Min1 = 1,
    Min5 = 5,
    Min15 = 15,
    Min30 = 30,
    Hour1 = 60,
    Hour4 = 240,
    Day1 = 1440,
}

impl SyncTimeframe {
    /// All timeframes in order from finest to coarsest.
    pub fn all() -> &'static [SyncTimeframe] {
        &[
            SyncTimeframe::Min1,
            SyncTimeframe::Min5,
            SyncTimeframe::Min15,
            SyncTimeframe::Min30,
            SyncTimeframe::Hour1,
            SyncTimeframe::Hour4,
            SyncTimeframe::Day1,
        ]
    }

    /// Convert to minutes.
    pub fn minutes(&self) -> i64 {
        *self as i64
    }

    /// Convert to milliseconds.
    pub fn millis(&self) -> i64 {
        self.minutes() * 60 * 1000
    }

    /// Get the MEXC API interval string.
    pub fn mexc_interval(&self) -> &'static str {
        match self {
            SyncTimeframe::Min1 => "1m",
            SyncTimeframe::Min5 => "5m",
            SyncTimeframe::Min15 => "15m",
            SyncTimeframe::Min30 => "30m",
            SyncTimeframe::Hour1 => "1h",
            SyncTimeframe::Hour4 => "4h",
            SyncTimeframe::Day1 => "1d",
        }
    }
}

/// DuckDB wrapper for candle storage.
pub struct CandleDb {
    conn: Connection,
}

impl CandleDb {
    /// Create or open a DuckDB database at the given path.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Ensure parent directory exists
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory: {:?}", parent))?;
        }

        let conn = Connection::open(path.as_ref())
            .with_context(|| format!("Failed to open database: {:?}", path.as_ref()))?;

        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Initialize the database schema.
    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL,
                timeframe INTEGER NOT NULL,
                timestamp BIGINT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                PRIMARY KEY (symbol, timeframe, timestamp)
            );

            CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_ts
                ON candles(symbol, timeframe, timestamp DESC);
            "#,
        )?;
        Ok(())
    }

    /// Insert candles into the database for a specific timeframe (upserts on conflict).
    pub fn insert_candles(
        &self,
        symbol: &str,
        timeframe: SyncTimeframe,
        candles: &[Candle],
    ) -> Result<usize> {
        if candles.is_empty() {
            return Ok(0);
        }

        let tf_minutes = timeframe.minutes();
        let mut stmt = self.conn.prepare(
            r#"
            INSERT OR REPLACE INTO candles (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            "#,
        )?;

        let mut count = 0;
        for candle in candles {
            // Convert timestamp from f64 (seconds) to i64 (milliseconds)
            let ts_ms = (candle.timestamp * 1000.0) as i64;
            stmt.execute(params![
                symbol,
                tf_minutes,
                ts_ms,
                candle.open as f64,
                candle.high as f64,
                candle.low as f64,
                candle.close as f64,
                candle.volume as f64,
            ])?;
            count += 1;
        }

        Ok(count)
    }

    /// Get the oldest timestamp for a symbol and timeframe (in milliseconds).
    pub fn get_oldest_timestamp(
        &self,
        symbol: &str,
        timeframe: SyncTimeframe,
    ) -> Result<Option<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT MIN(timestamp) FROM candles WHERE symbol = ? AND timeframe = ?",
        )?;
        let result: Option<i64> = stmt
            .query_row(params![symbol, timeframe.minutes()], |row| row.get(0))
            .ok();
        Ok(result)
    }

    /// Get the newest timestamp for a symbol and timeframe (in milliseconds).
    pub fn get_newest_timestamp(
        &self,
        symbol: &str,
        timeframe: SyncTimeframe,
    ) -> Result<Option<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT MAX(timestamp) FROM candles WHERE symbol = ? AND timeframe = ?",
        )?;
        let result: Option<i64> = stmt
            .query_row(params![symbol, timeframe.minutes()], |row| row.get(0))
            .ok();
        Ok(result)
    }

    /// Get the oldest timestamp across ALL timeframes for a symbol.
    pub fn get_oldest_timestamp_any(&self, symbol: &str) -> Result<Option<i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT MIN(timestamp) FROM candles WHERE symbol = ?")?;
        let result: Option<i64> = stmt.query_row([symbol], |row| row.get(0)).ok();
        Ok(result)
    }

    /// Get the newest timestamp across ALL timeframes for a symbol.
    pub fn get_newest_timestamp_any(&self, symbol: &str) -> Result<Option<i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT MAX(timestamp) FROM candles WHERE symbol = ?")?;
        let result: Option<i64> = stmt.query_row([symbol], |row| row.get(0)).ok();
        Ok(result)
    }

    /// Get the total candle count for a symbol across all timeframes.
    pub fn get_candle_count(&self, symbol: &str) -> Result<u64> {
        let mut stmt = self
            .conn
            .prepare("SELECT COUNT(*) FROM candles WHERE symbol = ?")?;
        let count: u64 = stmt.query_row([symbol], |row| row.get(0))?;
        Ok(count)
    }

    /// Get candle count for a specific timeframe.
    pub fn get_candle_count_for_timeframe(
        &self,
        symbol: &str,
        timeframe: SyncTimeframe,
    ) -> Result<u64> {
        let mut stmt = self
            .conn
            .prepare("SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?")?;
        let count: u64 = stmt.query_row(params![symbol, timeframe.minutes()], |row| row.get(0))?;
        Ok(count)
    }

    /// Load candles for a symbol, merging all timeframes.
    ///
    /// Returns candles from all timeframes, merged so that:
    /// - Finer resolution data takes priority over coarser data for overlapping periods
    /// - No synthetic/expanded candles are created
    /// - Each candle is at its native resolution
    ///
    /// The returned candles can be aggregated to any display timeframe.
    /// For display timeframes coarser than the data, aggregation works correctly.
    /// For display timeframes finer than the data, gaps will appear (as expected).
    pub fn load_candles(&self, symbol: &str) -> Result<Vec<Candle>> {
        // Track which time periods are covered by finer timeframes
        // Key: timestamp in ms (start of period), Value: (timeframe_minutes, candle)
        let mut covered_periods: BTreeMap<i64, (i64, Candle)> = BTreeMap::new();

        // Process timeframes from finest to coarsest
        // Finer timeframes mark their periods as "covered"
        for &tf in SyncTimeframe::all() {
            let tf_minutes = tf.minutes();
            let tf_millis = tf.millis();

            let mut stmt = self.conn.prepare(
                "SELECT timestamp, open, high, low, close, volume
                 FROM candles
                 WHERE symbol = ? AND timeframe = ?
                 ORDER BY timestamp ASC",
            )?;

            let rows = stmt.query_map(params![symbol, tf_minutes], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, f64>(2)?,
                    row.get::<_, f64>(3)?,
                    row.get::<_, f64>(4)?,
                    row.get::<_, f64>(5)?,
                ))
            })?;

            for row in rows {
                let (ts_ms, open, high, low, close, volume) = row?;

                // Check if this period is already covered by finer data
                let period_start = ts_ms;
                let period_end = ts_ms + tf_millis;

                // For coarser timeframes, check if ANY finer data exists in this period
                let has_finer_data = if tf_minutes > 1 {
                    covered_periods.range(period_start..period_end).any(|(_, (existing_tf, _))| {
                        *existing_tf < tf_minutes
                    })
                } else {
                    false
                };

                if !has_finer_data {
                    let candle = Candle {
                        timestamp: ts_ms as f64 / 1000.0,
                        open: open as f32,
                        high: high as f32,
                        low: low as f32,
                        close: close as f32,
                        volume: volume as f32,
                    };
                    covered_periods.insert(period_start, (tf_minutes, candle));
                }
            }
        }

        // Extract candles, removing any coarse candles that overlap with finer data
        // This handles edge cases where coarse candle was inserted before finer data arrived
        let mut result: Vec<Candle> = Vec::new();
        let mut skip_until: i64 = 0;

        for (ts, (tf_minutes, candle)) in &covered_periods {
            // Skip if this timestamp is within a period we're skipping
            if *ts < skip_until {
                continue;
            }

            result.push(candle.clone());

            // Mark the period this candle covers
            let tf_millis = tf_minutes * 60 * 1000;
            skip_until = ts + tf_millis;
        }

        Ok(result)
    }

    /// Load candles for a specific timeframe only.
    pub fn load_candles_for_timeframe(
        &self,
        symbol: &str,
        timeframe: SyncTimeframe,
    ) -> Result<Vec<Candle>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, open, high, low, close, volume
             FROM candles
             WHERE symbol = ? AND timeframe = ?
             ORDER BY timestamp ASC",
        )?;

        let candles: Vec<Candle> = stmt
            .query_map(params![symbol, timeframe.minutes()], |row| {
                let ts_ms: i64 = row.get(0)?;
                Ok(Candle {
                    timestamp: ts_ms as f64 / 1000.0,
                    open: row.get::<_, f64>(1)? as f32,
                    high: row.get::<_, f64>(2)? as f32,
                    low: row.get::<_, f64>(3)? as f32,
                    close: row.get::<_, f64>(4)? as f32,
                    volume: row.get::<_, f64>(5)? as f32,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(candles)
    }

    /// Find gaps in the candle data for a specific timeframe.
    pub fn find_gaps(&self, symbol: &str, timeframe: SyncTimeframe) -> Result<Vec<(i64, i64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp FROM candles WHERE symbol = ? AND timeframe = ? ORDER BY timestamp ASC",
        )?;

        let timestamps: Vec<i64> = stmt
            .query_map(params![symbol, timeframe.minutes()], |row| row.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut gaps = Vec::new();
        let tf_millis = timeframe.millis();
        let max_allowed_gap = tf_millis + 1000; // Allow 1 second tolerance

        for window in timestamps.windows(2) {
            let prev = window[0];
            let curr = window[1];
            let diff = curr - prev;

            if diff > max_allowed_gap {
                gaps.push((prev + tf_millis, curr - tf_millis));
            }
        }

        Ok(gaps)
    }

    /// Load candles within a time range for a specific timeframe.
    pub fn load_candles_range(
        &self,
        symbol: &str,
        timeframe: SyncTimeframe,
        start_ms: i64,
        end_ms: i64,
    ) -> Result<Vec<Candle>> {
        let mut stmt = self.conn.prepare(
            "SELECT timestamp, open, high, low, close, volume FROM candles
             WHERE symbol = ? AND timeframe = ? AND timestamp >= ? AND timestamp <= ?
             ORDER BY timestamp ASC",
        )?;

        let candles: Vec<Candle> = stmt
            .query_map(params![symbol, timeframe.minutes(), start_ms, end_ms], |row| {
                let ts_ms: i64 = row.get(0)?;
                Ok(Candle {
                    timestamp: ts_ms as f64 / 1000.0,
                    open: row.get::<_, f64>(1)? as f32,
                    high: row.get::<_, f64>(2)? as f32,
                    low: row.get::<_, f64>(3)? as f32,
                    close: row.get::<_, f64>(4)? as f32,
                    volume: row.get::<_, f64>(5)? as f32,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(candles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn now_seconds() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
    }

    #[test]
    fn test_db_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.duckdb");

        let db = CandleDb::open(&db_path).unwrap();

        let now = now_seconds();
        let candles = vec![
            Candle::new(now - 120.0, 100.0, 105.0, 99.0, 104.0, 1000.0),
            Candle::new(now - 60.0, 104.0, 108.0, 103.0, 107.0, 1500.0),
            Candle::new(now, 107.0, 110.0, 106.0, 109.0, 2000.0),
        ];

        let inserted = db.insert_candles("BTCUSDT", SyncTimeframe::Min1, &candles).unwrap();
        assert_eq!(inserted, 3);

        let count = db.get_candle_count("BTCUSDT").unwrap();
        assert_eq!(count, 3);

        let loaded = db.load_candles("BTCUSDT").unwrap();
        assert_eq!(loaded.len(), 3);
        assert!((loaded[0].open - 100.0).abs() < 0.01);
        assert!((loaded[2].close - 109.0).abs() < 0.01);
    }

    #[test]
    fn test_multi_timeframe_merge() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.duckdb");

        let db = CandleDb::open(&db_path).unwrap();

        // Base timestamp (aligned to 5 minutes for clean testing)
        let base_ts = 1700000000.0; // This should be aligned

        // Insert 1m candles for recent period
        let candles_1m = vec![
            Candle::new(base_ts, 100.0, 101.0, 99.0, 100.5, 100.0),
            Candle::new(base_ts + 60.0, 100.5, 102.0, 100.0, 101.5, 110.0),
        ];
        db.insert_candles("TEST", SyncTimeframe::Min1, &candles_1m).unwrap();

        // Insert 5m candle for OLDER period (before the 1m data)
        let candles_5m = vec![
            Candle::new(base_ts - 300.0, 98.0, 100.0, 97.0, 99.5, 500.0),
        ];
        db.insert_candles("TEST", SyncTimeframe::Min5, &candles_5m).unwrap();

        // load_candles() should return: 1 (5m) + 2 (1m) = 3 candles
        let loaded = db.load_candles("TEST").unwrap();
        assert_eq!(loaded.len(), 3);

        // First should be the 5m candle (older)
        assert!((loaded[0].open - 98.0).abs() < 0.01);
        // Then the 1m candles
        assert!((loaded[1].open - 100.0).abs() < 0.01);
        assert!((loaded[2].open - 100.5).abs() < 0.01);
    }

    #[test]
    fn test_finer_data_takes_priority() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.duckdb");

        let db = CandleDb::open(&db_path).unwrap();

        let base_ts = 1700000000.0;

        // Insert 5m candle
        let candles_5m = vec![
            Candle::new(base_ts, 98.0, 100.0, 97.0, 99.5, 500.0),
        ];
        db.insert_candles("TEST", SyncTimeframe::Min5, &candles_5m).unwrap();

        // Insert 1m candles that overlap with the 5m candle
        let candles_1m = vec![
            Candle::new(base_ts, 100.0, 101.0, 99.0, 100.5, 100.0),
            Candle::new(base_ts + 60.0, 100.5, 102.0, 100.0, 101.5, 110.0),
            Candle::new(base_ts + 120.0, 101.5, 103.0, 101.0, 102.5, 120.0),
        ];
        db.insert_candles("TEST", SyncTimeframe::Min1, &candles_1m).unwrap();

        // load_candles() should return only the 1m candles (finer takes priority)
        let loaded = db.load_candles("TEST").unwrap();
        assert_eq!(loaded.len(), 3);
        assert!((loaded[0].open - 100.0).abs() < 0.01); // 1m, not 5m's 98.0
    }
}

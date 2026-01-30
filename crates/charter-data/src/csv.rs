//! CSV data loading implementation.

use std::collections::HashMap;
use std::path::Path;

use charter_core::Candle;

use crate::DataSource;

/// Loads candle data from CSV files.
pub struct CsvLoader {
    path: std::path::PathBuf,
}

impl CsvLoader {
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

impl DataSource for CsvLoader {
    fn load(&self) -> anyhow::Result<Vec<Candle>> {
        load_candles_from_csv(&self.path)
    }
}

/// Parse datetime string "YYYY-MM-DD HH:MM:SS" or Unix timestamp to unix timestamp.
pub fn parse_datetime(s: &str) -> Option<f64> {
    // First try parsing as Unix timestamp (e.g., "1325412060.0")
    if let Ok(ts) = s.parse::<f64>() {
        return Some(ts);
    }

    // Format: "2017-08-17 04:00:00"
    let parts: Vec<&str> = s.split(&['-', ' ', ':']).collect();
    if parts.len() < 6 {
        return None;
    }
    let year: i32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let day: u32 = parts[2].parse().ok()?;
    let hour: u32 = parts[3].parse().ok()?;
    let min: u32 = parts[4].parse().ok()?;
    let sec: u32 = parts[5].parse().ok()?;

    // Simple timestamp calculation (not accounting for leap seconds, etc.)
    // Days since Unix epoch (1970-01-01)
    let mut days: i64 = 0;
    for y in 1970..year {
        days += if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) {
            366
        } else {
            365
        };
    }
    let month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    days += month_days[month as usize - 1] as i64;
    if month > 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) {
        days += 1;
    }
    days += (day - 1) as i64;

    let timestamp = days * 86400 + hour as i64 * 3600 + min as i64 * 60 + sec as i64;
    Some(timestamp as f64)
}

/// Load candles from CSV file and analyze for missing data.
/// Supports multiple formats:
/// - Format 1: Timestamp,Open,High,Low,Close,Volume (BTC style)
/// - Format 2: Unix Timestamp,Date,Symbol,Open,High,Low,Close,Volume (ETH style)
pub fn load_candles_from_csv<P: AsRef<Path>>(path: P) -> anyhow::Result<Vec<Candle>> {
    let mut reader = csv::ReaderBuilder::new().delimiter(b',').from_path(path)?;

    // Detect format from headers
    let headers = reader.headers()?.clone();
    let headers_lower: Vec<String> = headers.iter().map(|h| h.to_lowercase()).collect();

    // Find column indices
    let ts_col = headers_lower
        .iter()
        .position(|h| h.contains("timestamp") || h == "time");
    let open_col = headers_lower.iter().position(|h| h == "open");
    let high_col = headers_lower.iter().position(|h| h == "high");
    let low_col = headers_lower.iter().position(|h| h == "low");
    let close_col = headers_lower.iter().position(|h| h == "close");
    let volume_col = headers_lower.iter().position(|h| h == "volume");

    // Default to standard format if not found
    let ts_col = ts_col.unwrap_or(0);
    let open_col = open_col.unwrap_or(1);
    let high_col = high_col.unwrap_or(2);
    let low_col = low_col.unwrap_or(3);
    let close_col = close_col.unwrap_or(4);
    let volume_col = volume_col.unwrap_or(5);

    let mut candles = Vec::new();

    for result in reader.records() {
        let record = result?;

        let datetime_str = record.get(ts_col).unwrap_or("");
        let mut timestamp = parse_datetime(datetime_str).unwrap_or(0.0);

        // Detect milliseconds (13+ digits) vs seconds (10 digits)
        if timestamp > 1e12 {
            timestamp /= 1000.0;
        }

        let open: f32 = record.get(open_col).unwrap_or("0").parse()?;
        let high: f32 = record.get(high_col).unwrap_or("0").parse()?;
        let low: f32 = record.get(low_col).unwrap_or("0").parse()?;
        let close: f32 = record.get(close_col).unwrap_or("0").parse()?;
        let volume: f32 = record.get(volume_col).unwrap_or("0").parse()?;

        candles.push(Candle::new(timestamp, open, high, low, close, volume));
    }

    // Analyze for missing data points
    let timestamps: Vec<f64> = candles.iter().map(|c| c.timestamp).collect();
    analyze_data_gaps(&timestamps);

    // Sort by timestamp to ensure chronological order
    candles.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());

    Ok(candles)
}

/// Analyze data for gaps and missing data points.
pub fn analyze_data_gaps(timestamps: &[f64]) {
    if timestamps.len() < 2 {
        println!("Not enough data points to analyze gaps");
        return;
    }

    // Detect the interval (should be 60 seconds for minute data)
    let mut intervals: HashMap<i64, usize> = HashMap::new();
    for window in timestamps.windows(2) {
        let diff = (window[1] - window[0]).round() as i64;
        *intervals.entry(diff).or_insert(0) += 1;
    }

    // Find the most common interval (expected interval)
    let expected_interval = intervals
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(interval, _)| *interval)
        .unwrap_or(60);

    // Count gaps and missing data points
    let mut total_gaps = 0;
    let mut total_missing = 0;
    let mut largest_gap = 0i64;
    let mut largest_gap_start = 0f64;

    for window in timestamps.windows(2) {
        let diff = (window[1] - window[0]).round() as i64;
        if diff > expected_interval {
            total_gaps += 1;
            let missing_in_gap = (diff / expected_interval) - 1;
            total_missing += missing_in_gap;

            if diff > largest_gap {
                largest_gap = diff;
                largest_gap_start = window[0];
            }
        }
    }

    // Calculate time span
    let first_ts = timestamps.first().unwrap_or(&0.0);
    let last_ts = timestamps.last().unwrap_or(&0.0);
    let total_span_seconds = last_ts - first_ts;
    let expected_points = (total_span_seconds / expected_interval as f64).round() as i64;
    let actual_points = timestamps.len() as i64;
    let coverage_pct = (actual_points as f64 / expected_points as f64) * 100.0;

    println!("=== Data Analysis ===");
    println!("Total data points: {}", timestamps.len());
    println!("Expected interval: {} seconds", expected_interval);
    println!("Time span: {:.1} days", total_span_seconds / 86400.0);
    println!("Expected data points: {}", expected_points);
    println!(
        "Missing data points: {} ({:.2}% coverage)",
        total_missing, coverage_pct
    );
    println!("Number of gaps: {}", total_gaps);
    if largest_gap > 0 {
        println!(
            "Largest gap: {} seconds ({:.1} hours) at timestamp {}",
            largest_gap,
            largest_gap as f64 / 3600.0,
            largest_gap_start
        );
    }
    println!("=====================");
}

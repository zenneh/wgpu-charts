//! Example: Export ML training features from candle data.
//!
//! Usage:
//!   cargo run --example export_features -p charter-ta -- <input.csv> <output.csv> [lookahead] [sample_rate]
//!
//! Example:
//!   cargo run --example export_features -p charter-ta -- data/btc.csv data/training_data.csv 5 10
//!
//! Arguments:
//!   input.csv    - Input candle data (1-minute candles)
//!   output.csv   - Output training data file
//!   lookahead    - Candles to look ahead for direction label (default: 5)
//!   sample_rate  - Export every Nth candle (default: 1 = all candles)

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use charter_core::{Candle, Timeframe};
use charter_ta::{export_features_csv_with_config, ExportConfig};

/// Parse datetime string like "2024-10-29 05:43:00" to Unix timestamp
fn parse_datetime(s: &str) -> Option<f64> {
    // Try direct f64 parse first (Unix timestamp)
    if let Ok(ts) = s.parse::<f64>() {
        return Some(ts);
    }

    // Try datetime format: "YYYY-MM-DD HH:MM:SS"
    let parts: Vec<&str> = s.split(|c| c == ' ' || c == '-' || c == ':').collect();
    if parts.len() >= 6 {
        let year: i64 = parts[0].parse().ok()?;
        let month: i64 = parts[1].parse().ok()?;
        let day: i64 = parts[2].parse().ok()?;
        let hour: i64 = parts[3].parse().ok()?;
        let min: i64 = parts[4].parse().ok()?;
        let sec: i64 = parts[5].split('.').next()?.parse().ok()?;

        // Simple Unix timestamp calculation (not accounting for leap years perfectly)
        let days_since_epoch = (year - 1970) * 365 + (year - 1969) / 4 // leap years
            + (month - 1) * 30 + day - 1;
        let timestamp = days_since_epoch * 86400 + hour * 3600 + min * 60 + sec;
        return Some(timestamp as f64);
    }

    None
}

fn load_candles_from_csv(path: &Path) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut candles = Vec::new();
    let mut errors = 0;

    for (i, line) in reader.lines().enumerate() {
        let line = line?;

        // Skip header
        if i == 0 && (line.contains("Timestamp") || line.contains("timestamp") || line.contains("Open")) {
            continue;
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 6 {
            continue;
        }

        // Parse timestamp (could be Unix timestamp or datetime string)
        let timestamp = match parse_datetime(parts[0].trim()) {
            Some(ts) => ts,
            None => {
                errors += 1;
                if errors < 5 {
                    eprintln!("Warning: Could not parse timestamp: {}", parts[0]);
                }
                continue;
            }
        };

        // Parse OHLCV
        let open: f32 = match parts[1].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let high: f32 = match parts[2].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let low: f32 = match parts[3].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let close: f32 = match parts[4].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let volume: f32 = parts[5].trim().parse().unwrap_or(0.0);

        candles.push(Candle::new(timestamp, open, high, low, close, volume));

        // Progress
        if i > 0 && i % 500000 == 0 {
            eprintln!("  Loaded {} candles...", i);
        }
    }

    if errors > 0 {
        eprintln!("Warning: {} lines had parse errors", errors);
    }

    // Sort by timestamp
    candles.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());

    Ok(candles)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input.csv> <output.csv> [lookahead] [max_candles] [threshold]", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  input.csv    - Input candle data (1-minute candles)");
        eprintln!("  output.csv   - Output training data file");
        eprintln!("  lookahead    - Candles to look ahead for direction label (default: 5)");
        eprintln!("  max_candles  - Maximum candles to use (default: 100000)");
        eprintln!("  threshold    - Min price move % for UP/DOWN label (default: 0.1 = 0.1%)");
        eprintln!();
        eprintln!("The threshold filters out small price movements. Only movements larger");
        eprintln!("than the threshold are labeled as UP or DOWN; neutral moves are skipped.");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);
    let lookahead: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let max_candles: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let threshold: f32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.1) / 100.0; // Convert from % to decimal

    println!("Loading candles from {:?}...", input_path);
    let mut candles = load_candles_from_csv(input_path)?;
    println!("Loaded {} candles", candles.len());

    if candles.is_empty() {
        eprintln!("No candles loaded!");
        std::process::exit(1);
    }

    // Limit candle count if needed (use most recent candles)
    if candles.len() > max_candles {
        let skip = candles.len() - max_candles;
        candles = candles.into_iter().skip(skip).collect();
        println!("Using {} most recent candles", candles.len());
    }

    // Configure export for the 7 timeframes
    let config = ExportConfig::default()
        .with_timeframes(vec![
            Timeframe::Min1,
            Timeframe::Min3,
            Timeframe::Min5,
            Timeframe::Min30,
            Timeframe::Hour1,
            Timeframe::Day1,
            Timeframe::Week1,
        ])
        .with_min_candles(Timeframe::Min1, 100)
        .with_min_candles(Timeframe::Min3, 50)
        .with_min_candles(Timeframe::Min5, 30)
        .with_min_candles(Timeframe::Min30, 10)
        .with_min_candles(Timeframe::Hour1, 5)
        .with_min_candles(Timeframe::Day1, 2)
        .with_min_candles(Timeframe::Week1, 1)
        .with_min_active_levels(0)
        .with_label_lookahead(lookahead)
        .with_numbered_features(true)
        .with_label_threshold(threshold);

    println!("Exporting features to {:?}...", output_path);
    println!("  Timeframes: 1m, 3m, 5m, 30m, 1h, 1d, 1w");
    println!("  Lookahead: {} candles", lookahead);
    println!("  Label threshold: {:.2}%", threshold * 100.0);
    println!("  NOTE: This may take a while for large datasets...");

    let start = std::time::Instant::now();
    let rows = export_features_csv_with_config(&candles, output_path, &config)?;
    let elapsed = start.elapsed();

    println!("Exported {} feature rows in {:.1}s", rows, elapsed.as_secs_f64());
    println!("Done!");

    Ok(())
}

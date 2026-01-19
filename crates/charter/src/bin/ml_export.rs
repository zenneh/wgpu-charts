//! ML Training Data Export Tool
//!
//! Processes CSV data and exports training samples for ML model training.
//! Uses recent data subset for efficient TA computation.
//!
//! Usage: charter-ml-export <csv_path> <output_path> [--lookahead N] [--max-candles N] [--step N]

use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use charter_core::{aggregate_candles, Candle, Timeframe};
use charter_data::load_candles_from_csv;
use charter_ta::{
    Analyzer, AnalyzerConfig, Level, LevelState, MlFeatures, TimeframeFeatures, TrainingDataset,
    TrainingSample, Trend,
};

/// Default lookahead candles for labeling (10 candles = 50 minutes for 5m).
const DEFAULT_LOOKAHEAD: usize = 10;

/// Default max candles to use from data (200k 5m candles = ~694 days).
const DEFAULT_MAX_CANDLES: usize = 200_000;

/// Default step between samples (sample every N candles).
const DEFAULT_STEP: usize = 10;

/// Minimum candles required for TA analysis.
const MIN_CANDLES: usize = 100;

/// Skip early candles to let TA patterns form.
const WARMUP_CANDLES: usize = 500;

/// Timeframes for ML training (4 total: 5m, 1h, 1d, 1w).
fn ml_timeframes() -> Vec<Timeframe> {
    vec![
        Timeframe::Min5,  // Short-term
        Timeframe::Hour1, // Medium-term
        Timeframe::Day1,  // Long-term
        Timeframe::Week1, // Very long-term
    ]
}

/// Temporary TA data holder for the exporter.
#[derive(Default)]
struct TimeframeTaData {
    levels: Vec<Level>,
    trends: Vec<Trend>,
    candle_count: usize,
}

fn main() -> Result<()> {
    env_logger::init();
    let start_time = Instant::now();

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <csv_path> <output_path> [options]", args[0]);
        eprintln!("Options:");
        eprintln!(
            "  --lookahead N    Candles to look ahead for label (default: {})",
            DEFAULT_LOOKAHEAD
        );
        eprintln!(
            "  --max-candles N  Use last N candles from 1m data (default: {})",
            DEFAULT_MAX_CANDLES
        );
        eprintln!(
            "  --step N         Sample every N candles (default: {})",
            DEFAULT_STEP
        );
        eprintln!();
        eprintln!(
            "Example: {} data.csv training.csv --max-candles 300000 --step 20",
            args[0]
        );
        std::process::exit(1);
    }

    let csv_path = &args[1];
    let output_path = &args[2];

    // Parse options
    let mut lookahead = DEFAULT_LOOKAHEAD;
    let mut max_candles = DEFAULT_MAX_CANDLES;
    let mut step = DEFAULT_STEP;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--lookahead" if i + 1 < args.len() => {
                lookahead = args[i + 1].parse().unwrap_or(DEFAULT_LOOKAHEAD);
                i += 2;
            }
            "--max-candles" if i + 1 < args.len() => {
                max_candles = args[i + 1].parse().unwrap_or(DEFAULT_MAX_CANDLES);
                i += 2;
            }
            "--step" if i + 1 < args.len() => {
                step = args[i + 1].parse().unwrap_or(DEFAULT_STEP);
                i += 2;
            }
            _ => i += 1,
        }
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           ML Training Data Export                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load base data (1m candles)
    print!("📂 Loading CSV data... ");
    io::stdout().flush()?;
    let load_start = Instant::now();
    let all_candles = load_candles_from_csv(csv_path)?;
    let total_loaded = all_candles.len();

    // Use only recent candles
    let base_1m: Vec<Candle> = if all_candles.len() > max_candles {
        let skip = all_candles.len() - max_candles;
        println!("using last {}k of {}", max_candles / 1000, total_loaded);
        all_candles[skip..].to_vec()
    } else {
        println!("✓ {} candles", all_candles.len());
        all_candles
    };
    println!("   Loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    // Aggregate to 5m as primary timeframe
    print!("📊 Aggregating to 5m... ");
    io::stdout().flush()?;
    let primary_candles = aggregate_candles(&base_1m, Timeframe::Min5);
    println!("✓ {} candles", primary_candles.len());

    println!();
    println!("⚙️  Configuration:");
    println!(
        "   • Primary: 5m candles ({} candles = ~{} days)",
        primary_candles.len(),
        primary_candles.len() / 288
    );
    println!("   • Step size: {} candles", step);
    println!("   • Lookahead: {} candles", lookahead);
    println!("   • Timeframes: 4 (5m, 1h, 1d, 1w)");
    println!();

    // Aggregate to all timeframes
    let timeframes = ml_timeframes();
    let mut timeframe_candles: Vec<Vec<Candle>> = Vec::new();

    println!("📊 Aggregating timeframes...");
    for (i, tf) in timeframes.iter().enumerate() {
        print!("   [{}/{}] {} ... ", i + 1, timeframes.len(), tf.label());
        io::stdout().flush()?;
        let tf_start = Instant::now();
        let candles = aggregate_candles(&base_1m, *tf);
        println!(
            "✓ {} candles ({:.2}s)",
            candles.len(),
            tf_start.elapsed().as_secs_f32()
        );
        timeframe_candles.push(candles);
    }

    // Run TA analysis for each timeframe
    println!();
    println!("📈 Running TA analysis...");
    let mut timeframe_data: Vec<TimeframeTaData> = Vec::new();

    for (tf_idx, candles) in timeframe_candles.iter().enumerate() {
        print!(
            "   [{}/{}] {} ... ",
            tf_idx + 1,
            timeframes.len(),
            timeframes[tf_idx].label()
        );
        io::stdout().flush()?;

        if candles.len() < MIN_CANDLES {
            println!(
                "⚠️  skipped (only {} candles, need {})",
                candles.len(),
                MIN_CANDLES
            );
            timeframe_data.push(TimeframeTaData::default());
            continue;
        }

        let ta_start = Instant::now();
        let config = AnalyzerConfig::default();
        let mut analyzer = Analyzer::with_config(config);

        let total = candles.len();
        let mut last_pct = 0;
        for (i, candle) in candles.iter().enumerate() {
            analyzer.process_candle(*candle);

            // Progress indicator for large timeframes
            if total > 50_000 {
                let pct = (i * 100) / total;
                if pct > last_pct && pct % 20 == 0 {
                    last_pct = pct;
                    print!("{}%...", pct);
                    io::stdout().flush()?;
                }
            }
        }

        let data = TimeframeTaData {
            levels: analyzer.all_levels().to_vec(),
            trends: analyzer.all_trends().to_vec(),
            candle_count: candles.len(),
        };

        println!(
            "✓ {} levels, {} trends ({:.2}s)",
            data.levels.len(),
            data.trends.len(),
            ta_start.elapsed().as_secs_f32()
        );
        timeframe_data.push(data);
    }

    // Generate training samples using 5m as primary
    println!();
    println!("🔄 Generating training samples...");

    let start_idx = WARMUP_CANDLES;
    let end_idx = primary_candles.len().saturating_sub(lookahead);
    let total_samples = if end_idx > start_idx {
        (end_idx - start_idx) / step
    } else {
        0
    };

    let mut dataset = TrainingDataset::new(4, lookahead); // 4 timeframes
    let mut samples_created = 0;
    let sample_start = Instant::now();
    let mut last_progress = 0;

    for sample_idx in (start_idx..end_idx).step_by(step) {
        let current_candle = &primary_candles[sample_idx];
        let current_timestamp = current_candle.timestamp;

        // Extract features for all timeframes
        let mut tf_features: Vec<TimeframeFeatures> = Vec::new();

        for (tf_idx, (candles, ta_data)) in timeframe_candles
            .iter()
            .zip(timeframe_data.iter())
            .enumerate()
        {
            if ta_data.candle_count < MIN_CANDLES {
                continue;
            }

            // Find the corresponding candle index in this timeframe
            let tf_candle_idx = find_candle_index(candles, current_timestamp);
            if tf_candle_idx.is_none() {
                continue;
            }
            let tf_candle_idx = tf_candle_idx.unwrap();

            // Get levels and trends that existed at this point
            let active_levels: Vec<_> = ta_data
                .levels
                .iter()
                .filter(|l| l.created_at_index <= tf_candle_idx)
                .cloned()
                .collect();

            let active_trends: Vec<_> = ta_data
                .trends
                .iter()
                .filter(|t| t.created_at_index <= tf_candle_idx)
                .cloned()
                .collect();

            let features = TimeframeFeatures::extract(
                tf_idx,
                &active_levels,
                &active_trends,
                current_candle.close,
                tf_candle_idx,
            );
            tf_features.push(features);
        }

        if tf_features.is_empty() {
            continue;
        }

        // Calculate candle features
        let prev_close = if sample_idx > 0 {
            primary_candles[sample_idx - 1].close
        } else {
            current_candle.open
        };
        let price_change = if prev_close > 0.0 {
            (current_candle.close - prev_close) / prev_close
        } else {
            0.0
        };

        let body = (current_candle.close - current_candle.open).abs();
        let range = current_candle.high - current_candle.low;
        let body_ratio = if range > 0.0 { body / range } else { 0.5 };

        // Normalize volume
        let lookback = 100.min(sample_idx);
        let avg_volume: f32 = primary_candles[sample_idx - lookback..sample_idx]
            .iter()
            .map(|c| c.volume)
            .sum::<f32>()
            / lookback as f32;
        let volume_normalized = if avg_volume > 0.0 {
            current_candle.volume / avg_volume
        } else {
            1.0
        };

        let ml_features = MlFeatures {
            timeframes: tf_features,
            current_price: current_candle.close,
            current_volume_normalized: volume_normalized,
            price_change_normalized: price_change,
            body_ratio,
            is_bullish: if current_candle.close > current_candle.open {
                1.0
            } else {
                0.0
            },
        };

        // Create sample with labels
        let mut sample =
            TrainingSample::new(ml_features, sample_idx, current_candle.timestamp, lookahead);

        // Get future candle for labeling
        let future_candle = &primary_candles[sample_idx + lookahead];

        // Check if nearest level broke
        let nearest_level_broke = check_nearest_level_broke(
            &timeframe_data[0], // Use 5m timeframe
            sample_idx,
            sample_idx + lookahead,
            current_candle.close,
        );

        sample.fill_labels(future_candle.close, nearest_level_broke);
        dataset.add_sample(sample);
        samples_created += 1;

        // Progress bar
        let progress = (samples_created * 100) / total_samples.max(1);
        if progress > last_progress {
            last_progress = progress;
            let elapsed = sample_start.elapsed().as_secs_f32();
            let rate = samples_created as f32 / elapsed;
            let remaining = (total_samples - samples_created) as f32 / rate;
            print!("\r   Progress: [");
            for p in 0..20 {
                if p < progress / 5 {
                    print!("█");
                } else {
                    print!("░");
                }
            }
            print!(
                "] {}% ({}/{}) - {:.0}/s - ETA: {:.0}s    ",
                progress, samples_created, total_samples, rate, remaining
            );
            io::stdout().flush()?;
        }
    }
    println!();

    let sample_time = sample_start.elapsed().as_secs_f32();
    println!(
        "   ✓ Created {} samples in {:.2}s ({:.0} samples/s)",
        samples_created,
        sample_time,
        samples_created as f32 / sample_time
    );

    // Export to CSV
    println!();
    println!("💾 Exporting training data...");
    let csv_path = Path::new(output_path).with_extension("csv");
    print!("   Writing CSV... ");
    io::stdout().flush()?;
    export_csv(&dataset, &csv_path)?;
    println!("✓ {}", csv_path.display());

    // Summary
    let total_time = start_time.elapsed().as_secs_f32();
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                        Summary                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("   • Total samples: {}", samples_created);
    println!(
        "   • Feature dimensions: {} (4 timeframes × 74 features + 5 global)",
        dataset.feature_dim
    );

    let up_count = dataset
        .labeled_samples()
        .filter(|s| s.direction_after_n == Some(1))
        .count();
    println!(
        "   • Direction up: {:.1}%",
        up_count as f32 / samples_created.max(1) as f32 * 100.0
    );
    println!("   • Total time: {:.2}s", total_time);
    println!();
    println!(
        "✨ Export complete! Train with: python scripts/ml/train_model.py {}",
        csv_path.display()
    );

    Ok(())
}

/// Find the candle index closest to a given timestamp.
fn find_candle_index(candles: &[Candle], timestamp: f64) -> Option<usize> {
    if candles.is_empty() {
        return None;
    }

    let mut lo = 0;
    let mut hi = candles.len() - 1;

    while lo < hi {
        let mid = (lo + hi) / 2;
        if candles[mid].timestamp < timestamp {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }

    if lo > 0
        && (candles[lo].timestamp - timestamp).abs()
            > (candles[lo - 1].timestamp - timestamp).abs()
    {
        Some(lo - 1)
    } else {
        Some(lo)
    }
}

/// Check if the nearest level to current price broke within the lookahead window.
fn check_nearest_level_broke(
    ta_data: &TimeframeTaData,
    start_idx: usize,
    end_idx: usize,
    current_price: f32,
) -> bool {
    let nearest = ta_data
        .levels
        .iter()
        .filter(|l| l.created_at_index <= start_idx && l.state != LevelState::Broken)
        .min_by(|a, b| {
            let dist_a = (a.price - current_price).abs();
            let dist_b = (b.price - current_price).abs();
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        });

    if let Some(level) = nearest {
        if let Some(break_event) = &level.break_event {
            return break_event.candle_index > start_idx && break_event.candle_index <= end_idx;
        }
    }

    false
}

/// Export dataset to CSV format.
fn export_csv(dataset: &TrainingDataset, path: &Path) -> Result<()> {
    let mut file = File::create(path)?;

    let (features, labels) = dataset.to_numpy_format();
    if features.is_empty() {
        return Ok(());
    }

    let feature_count = features[0].len();
    let mut header = String::new();
    for i in 0..feature_count {
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!("f{}", i));
    }
    header.push_str(",level_broke,direction_up,price_change_pct\n");
    file.write_all(header.as_bytes())?;

    for (feat, label) in features.iter().zip(labels.iter()) {
        let mut row = String::new();
        for (i, f) in feat.iter().enumerate() {
            if i > 0 {
                row.push(',');
            }
            row.push_str(&format!("{:.6}", f));
        }
        row.push_str(&format!(
            ",{:.0},{:.0},{:.4}\n",
            label[0], label[1], label[2]
        ));
        file.write_all(row.as_bytes())?;
    }

    Ok(())
}

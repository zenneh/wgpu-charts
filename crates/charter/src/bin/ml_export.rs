//! Fast ML Training Data Export Tool
//!
//! Processes CSV data and exports training samples for ML model training.
//! Uses incremental multi-timeframe analysis for maximum speed.
//!
//! Usage: charter-ml-export <csv_path> <output_path> [--lookahead N] [--max-candles N] [--step N]

use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use charter_core::Candle;
use charter_data::load_candles_from_csv;
use charter_ta::{
    ml_export_timeframes, AnalyzerConfig, LevelState, MlFeatures, MultiTimeframeAnalyzer,
    TimeframeFeatures,
};

/// Default lookahead candles for labeling (10 candles at 1m = 10 minutes).
const DEFAULT_LOOKAHEAD: usize = 10;

/// Default max candles to use from data (500k 1m candles = ~347 days).
const DEFAULT_MAX_CANDLES: usize = 500_000;

/// Default step between samples (sample every N 1m candles).
const DEFAULT_STEP: usize = 5;

/// Minimum candles required before sampling (warmup).
const WARMUP_CANDLES: usize = 1000;

/// RSI period.
const RSI_PERIOD: usize = 14;

/// Fast RSI calculation using Wilder's smoothing.
struct RsiCalculator {
    period: usize,
    avg_gain: f32,
    avg_loss: f32,
    prev_close: f32,
    count: usize,
    gains: Vec<f32>,
    losses: Vec<f32>,
}

impl RsiCalculator {
    fn new(period: usize) -> Self {
        Self {
            period,
            avg_gain: 0.0,
            avg_loss: 0.0,
            prev_close: 0.0,
            count: 0,
            gains: Vec::with_capacity(period),
            losses: Vec::with_capacity(period),
        }
    }

    #[inline]
    fn update(&mut self, close: f32) -> f32 {
        if self.count == 0 {
            self.prev_close = close;
            self.count = 1;
            return 0.5; // Neutral
        }

        let change = close - self.prev_close;
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };
        self.prev_close = close;

        if self.count <= self.period {
            // Initial period - accumulate
            self.gains.push(gain);
            self.losses.push(loss);
            self.count += 1;

            if self.count == self.period + 1 {
                // Calculate initial averages
                self.avg_gain = self.gains.iter().sum::<f32>() / self.period as f32;
                self.avg_loss = self.losses.iter().sum::<f32>() / self.period as f32;
                // Clear storage - no longer needed
                self.gains.clear();
                self.losses.clear();
            } else {
                return 0.5; // Not enough data yet
            }
        } else {
            // Wilder's smoothing
            self.avg_gain = (self.avg_gain * (self.period - 1) as f32 + gain) / self.period as f32;
            self.avg_loss = (self.avg_loss * (self.period - 1) as f32 + loss) / self.period as f32;
            self.count += 1;
        }

        // Calculate RSI
        if self.avg_loss == 0.0 {
            return 1.0;
        }
        let rs = self.avg_gain / self.avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        rsi / 100.0 // Normalize to 0-1
    }
}

/// Rolling volume average calculator.
struct VolumeNormalizer {
    window: Vec<f32>,
    window_size: usize,
    sum: f32,
    idx: usize,
    filled: bool,
}

impl VolumeNormalizer {
    fn new(window_size: usize) -> Self {
        Self {
            window: vec![0.0; window_size],
            window_size,
            sum: 0.0,
            idx: 0,
            filled: false,
        }
    }

    #[inline]
    fn update(&mut self, volume: f32) -> f32 {
        // Remove old value from sum
        self.sum -= self.window[self.idx];
        // Add new value
        self.window[self.idx] = volume;
        self.sum += volume;

        self.idx = (self.idx + 1) % self.window_size;
        if self.idx == 0 {
            self.filled = true;
        }

        let count = if self.filled {
            self.window_size
        } else {
            self.idx.max(1)
        };
        let avg = self.sum / count as f32;

        if avg > 0.0 {
            volume / avg
        } else {
            1.0
        }
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let start_time = Instant::now();

    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <csv_path> <output_path> [options]", args[0]);
        eprintln!("Options:");
        eprintln!(
            "  --lookahead N    1m candles to look ahead for label (default: {})",
            DEFAULT_LOOKAHEAD
        );
        eprintln!(
            "  --max-candles N  Use last N 1m candles (default: {})",
            DEFAULT_MAX_CANDLES
        );
        eprintln!(
            "  --step N         Sample every N 1m candles (default: {})",
            DEFAULT_STEP
        );
        eprintln!();
        eprintln!(
            "Example: {} data.csv training.csv --max-candles 300000 --step 10",
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
    println!("║        Fast ML Training Data Export (Incremental TA)         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load base data (1m candles)
    print!("📂 Loading CSV data... ");
    io::stdout().flush()?;
    let load_start = Instant::now();
    let all_candles = load_candles_from_csv(csv_path)?;
    let total_loaded = all_candles.len();

    // Use only recent candles
    let candles_1m: Vec<Candle> = if all_candles.len() > max_candles {
        let skip = all_candles.len() - max_candles;
        println!("using last {}k of {}", max_candles / 1000, total_loaded);
        all_candles[skip..].to_vec()
    } else {
        println!("✓ {} candles", all_candles.len());
        all_candles
    };
    println!("   Loaded in {:.2}s", load_start.elapsed().as_secs_f32());

    let timeframes = ml_export_timeframes();
    let num_timeframes = timeframes.len();

    println!();
    println!("⚙️  Configuration:");
    println!(
        "   • 1m candles: {} (~{} days)",
        candles_1m.len(),
        candles_1m.len() / 1440
    );
    println!("   • Step size: {} (1m candles)", step);
    println!("   • Lookahead: {} (1m candles)", lookahead);
    println!(
        "   • Timeframes: {} ({:?})",
        num_timeframes,
        timeframes.iter().map(|t| t.label()).collect::<Vec<_>>()
    );
    println!();

    // Create incremental multi-timeframe analyzer
    let config = AnalyzerConfig::default();
    let mut mta = MultiTimeframeAnalyzer::new(timeframes.clone(), config);

    // Create calculators
    let mut rsi = RsiCalculator::new(RSI_PERIOD);
    let mut vol_norm = VolumeNormalizer::new(100);

    // Prepare for sampling
    let total_candles = candles_1m.len();
    let end_idx = total_candles.saturating_sub(lookahead);
    let start_idx = WARMUP_CANDLES;
    let _expected_samples = if end_idx > start_idx {
        (end_idx - start_idx) / step
    } else {
        0
    };

    println!("🔄 Processing {} 1m candles with incremental TA...", total_candles);
    let process_start = Instant::now();

    // Prepare output file with buffered writer
    let csv_out_path = Path::new(output_path).with_extension("csv");
    let file = File::create(&csv_out_path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file); // 1MB buffer

    // Write header
    let feature_count = num_timeframes * 74 + 6;
    let mut header = String::with_capacity(feature_count * 4);
    for i in 0..feature_count {
        if i > 0 {
            header.push(',');
        }
        header.push_str(&format!("f{}", i));
    }
    header.push_str(",level_broke,direction_up,price_change_pct\n");
    writer.write_all(header.as_bytes())?;

    let mut samples_written = 0;
    let mut last_progress = 0;
    let mut prev_close = candles_1m.first().map(|c| c.close).unwrap_or(0.0);

    // Process candles incrementally
    for (idx, candle) in candles_1m.iter().enumerate() {
        // Update technical analysis
        mta.process_1m_candle(candle);

        // Update indicators
        let rsi_value = rsi.update(candle.close);
        let vol_normalized = vol_norm.update(candle.volume);

        // Check if we should sample at this point
        let should_sample = idx >= start_idx && idx < end_idx && (idx - start_idx) % step == 0;

        if should_sample {
            // Extract features from current TA state
            let tf_features: Vec<TimeframeFeatures> = mta.extract_all_features(candle.close);

            if !tf_features.is_empty() {
                // Calculate candle features
                let price_change = if prev_close > 0.0 {
                    (candle.close - prev_close) / prev_close
                } else {
                    0.0
                };

                let body = (candle.close - candle.open).abs();
                let range = candle.high - candle.low;
                let body_ratio = if range > 0.0 { body / range } else { 0.5 };

                // Get future candle for labeling
                let future_candle = &candles_1m[idx + lookahead];
                let price_change_pct = (future_candle.close - candle.close) / candle.close * 100.0;
                let direction_up = if future_candle.close > candle.close { 1 } else { 0 };

                // Check if nearest level broke (use first timeframe - 1m)
                let level_broke = check_nearest_level_will_break(&mta, idx, idx + lookahead, candle.close);

                // Build feature vector
                let ml_features = MlFeatures {
                    timeframes: tf_features,
                    current_price: candle.close,
                    current_volume_normalized: vol_normalized,
                    price_change_normalized: price_change,
                    body_ratio,
                    is_bullish: if candle.close > candle.open { 1.0 } else { 0.0 },
                    rsi_14: rsi_value,
                };

                // Write directly to CSV (skip TrainingSample overhead)
                let features = ml_features.to_vec();
                let mut row = String::with_capacity(features.len() * 10);
                for (i, f) in features.iter().enumerate() {
                    if i > 0 {
                        row.push(',');
                    }
                    row.push_str(&format!("{:.6}", f));
                }
                row.push_str(&format!(
                    ",{},{},{:.4}\n",
                    if level_broke { 1 } else { 0 },
                    direction_up,
                    price_change_pct
                ));
                writer.write_all(row.as_bytes())?;
                samples_written += 1;
            }
        }

        prev_close = candle.close;

        // Progress indicator
        let progress = (idx * 100) / total_candles;
        if progress > last_progress && progress % 5 == 0 {
            last_progress = progress;
            let elapsed = process_start.elapsed().as_secs_f32();
            let rate = idx as f32 / elapsed;
            let remaining = (total_candles - idx) as f32 / rate;
            print!(
                "\r   Progress: [{}{}] {}% - {:.0} candles/s - ETA: {:.0}s - {} samples    ",
                "█".repeat(progress / 5),
                "░".repeat(20 - progress / 5),
                progress,
                rate,
                remaining,
                samples_written
            );
            io::stdout().flush()?;
        }
    }

    // Flush remaining data
    writer.flush()?;
    println!();

    let process_time = process_start.elapsed().as_secs_f32();
    println!(
        "   ✓ Processed {} candles in {:.2}s ({:.0} candles/s)",
        total_candles,
        process_time,
        total_candles as f32 / process_time
    );

    // Report TA statistics
    println!();
    println!("📊 TA Statistics:");
    for (tf, levels, trends) in mta.ta_counts() {
        println!("   • {}: {} levels, {} trends", tf.label(), levels, trends);
    }

    // Summary
    let total_time = start_time.elapsed().as_secs_f32();
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                        Summary                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("   • Total samples: {}", samples_written);
    println!(
        "   • Feature dimensions: {} ({} timeframes × 74 features + 6 global)",
        feature_count, num_timeframes
    );
    println!("   • Total time: {:.2}s", total_time);
    println!("   • Output: {}", csv_out_path.display());
    println!();
    println!(
        "✨ Export complete! Train with: python scripts/ml/tune_hyperparameters.py {} charter_model.onnx --trials 100",
        csv_out_path.display()
    );

    Ok(())
}

/// Check if the nearest level to current price will break within the lookahead window.
/// This is a simplified version that checks if price moved significantly through any nearby level.
fn check_nearest_level_will_break(
    mta: &MultiTimeframeAnalyzer,
    _start_idx: usize,
    _end_idx: usize,
    current_price: f32,
) -> bool {
    // Use the 1m timeframe (index 0) for level checking
    if let Some(state) = mta.state(0) {
        let levels = state.all_levels();

        // Find nearest active level
        let nearest = levels
            .iter()
            .filter(|l| l.state != LevelState::Broken)
            .min_by(|a, b| {
                let dist_a = (a.price - current_price).abs();
                let dist_b = (b.price - current_price).abs();
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(level) = nearest {
            // Check if level is close enough to potentially break
            let distance_pct = ((level.price - current_price) / current_price).abs();
            // If level is within 0.5% of current price, consider it at risk
            return distance_pct < 0.005;
        }
    }

    false
}

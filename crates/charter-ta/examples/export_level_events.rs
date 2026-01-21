//! Export level approach events for ML training.
//!
//! This exports data for training a model to predict whether a level will hold or break.
//!
//! Usage: cargo run --example export_level_events -p charter-ta --release -- \
//!     <input_csv> <output_csv> [max_candles] [lookahead]
//!
//! Example: cargo run --example export_level_events -p charter-ta --release -- \
//!     data/btc-new.csv data/level_events.csv 500000 5

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use charter_core::Timeframe;
use charter_data::load_candles_from_csv;
use charter_ta::{
    determine_outcome, extract_level_features, is_approaching_level,
    Analyzer, AnalyzerConfig, DefaultAnalyzer, LevelEventFeatures, LevelOutcome,
    TimeframeConfig, aggregate_candles,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <input_csv> <output_csv> [max_candles] [lookahead]", args[0]);
        eprintln!("  max_candles: Maximum candles to process (default: 500000)");
        eprintln!("  lookahead: Candles to look ahead for outcome (default: 5)");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let max_candles: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(500_000);
    let lookahead: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5);

    println!("Loading candles from {:?}...", input_path);
    let start = Instant::now();

    let all_candles = load_candles_from_csv(input_path)?;
    let candles_1m = if all_candles.len() > max_candles {
        all_candles[all_candles.len() - max_candles..].to_vec()
    } else {
        all_candles
    };

    println!("  Loaded {} candles in {:.1}s", candles_1m.len(), start.elapsed().as_secs_f32());

    // Setup timeframes
    let timeframes = vec![
        Timeframe::Min1,
        Timeframe::Min3,
        Timeframe::Min5,
        Timeframe::Min30,
        Timeframe::Hour1,
        Timeframe::Day1,
        Timeframe::Week1,
    ];

    // Pre-aggregate candles
    println!("Aggregating candles for {} timeframes...", timeframes.len());
    let mut timeframe_candles: HashMap<Timeframe, Vec<charter_core::Candle>> = HashMap::new();
    for &tf in &timeframes {
        let candles = if tf == Timeframe::Min1 {
            candles_1m.clone()
        } else {
            aggregate_candles(&candles_1m, tf)
        };
        println!("  {}: {} candles", tf.label(), candles.len());
        timeframe_candles.insert(tf, candles);
    }

    // Setup analyzer
    let tf_configs: Vec<TimeframeConfig> = timeframes
        .iter()
        .map(|&tf| TimeframeConfig::new(tf, 3, 0.001))
        .collect();
    let analyzer_config = AnalyzerConfig::new(tf_configs);
    let mut analyzer = DefaultAnalyzer::new(analyzer_config);

    // Open output file
    let file = File::create(output_path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    let feature_names = LevelEventFeatures::feature_names();
    writeln!(writer, "{},held", feature_names.join(","))?;

    // Track events
    let mut events_written = 0;
    let mut held_count = 0;
    let mut broke_count = 0;

    // Track which levels we've seen events for (to avoid duplicates)
    let mut processed_approaches: HashMap<(u64, usize), bool> = HashMap::new();

    // Pre-compute timeframe candle counts at each 1m index
    let mut tf_counts: HashMap<Timeframe, Vec<usize>> = HashMap::new();
    for &tf in &timeframes {
        let mut counts = Vec::with_capacity(candles_1m.len());
        if tf == Timeframe::Min1 {
            for i in 0..candles_1m.len() {
                counts.push(i + 1);
            }
        } else {
            let mut count = 0;
            let mut last_bucket: Option<i64> = None;
            for candle in &candles_1m {
                let bucket = (candle.timestamp / tf.seconds()).floor() as i64;
                if last_bucket.is_none() || bucket != last_bucket.unwrap() {
                    if last_bucket.is_some() {
                        count += 1;
                    }
                    last_bucket = Some(bucket);
                }
                counts.push(count);
            }
        }
        tf_counts.insert(tf, counts);
    }

    println!("Processing candles and extracting level events...");
    let process_start = Instant::now();

    let mut last_tf_counts: HashMap<Timeframe, usize> = timeframes.iter().map(|&tf| (tf, 0)).collect();

    for idx in 0..candles_1m.len() {
        let candle = &candles_1m[idx];
        let current_price = candle.close;

        // Update analyzer with new timeframe candles
        for (tf_idx, &tf) in timeframes.iter().enumerate() {
            let tf_candle_count = tf_counts[&tf][idx];
            let last_count = last_tf_counts[&tf];

            if tf_candle_count > last_count && tf_candle_count > 0 {
                let tf_candles = &timeframe_candles[&tf][..tf_candle_count];
                analyzer.update(tf_idx as u8, tf_candles, current_price);
                last_tf_counts.insert(tf, tf_candle_count);
            }
        }

        // Skip if not enough lookahead data
        if idx + lookahead >= candles_1m.len() {
            continue;
        }

        // Get analyzer state and check all levels
        let state = analyzer.state();

        for (_tf_idx, tf_state) in &state.timeframe_states {
            for level in tf_state.level_index.active_levels() {
                // Check if approaching this level
                if !is_approaching_level(level, current_price) {
                    continue;
                }

                // Skip if we already processed an event for this level recently
                let event_key = (level.id.0, idx / 10); // Group by 10-candle windows
                if processed_approaches.contains_key(&event_key) {
                    continue;
                }

                // Get recent candles for momentum calculation
                let momentum_start = idx.saturating_sub(5);
                let recent_candles = &candles_1m[momentum_start..idx];

                if recent_candles.len() < 3 {
                    continue;
                }

                // Count nearby levels
                let nearby = tf_state.level_index
                    .levels_in_range(current_price * 0.99, current_price * 1.01)
                    .len();

                // Get opposite level distance
                let opposite_distance = match level.level_direction {
                    charter_ta::types::LevelDirection::Support => {
                        tf_state.level_index.closest_resistance_above(current_price)
                            .map(|l| (l.price - current_price) / current_price)
                    }
                    charter_ta::types::LevelDirection::Resistance => {
                        tf_state.level_index.closest_support_below(current_price)
                            .map(|l| (current_price - l.price) / current_price)
                    }
                };

                // Extract features
                let features = extract_level_features(
                    level,
                    current_price,
                    candle,
                    recent_candles,
                    idx,
                    nearby,
                    opposite_distance,
                );

                // Determine outcome using subsequent candles
                let subsequent = &candles_1m[idx + 1..];
                let (outcome, _resolution_candles) = determine_outcome(
                    level,
                    candle,
                    subsequent,
                    lookahead,
                );

                // Only record events with clear outcomes
                let label = match outcome {
                    LevelOutcome::Held => {
                        held_count += 1;
                        1
                    }
                    LevelOutcome::Broke => {
                        broke_count += 1;
                        0
                    }
                    LevelOutcome::Pending => continue,
                };

                // Write features
                let feature_values: Vec<String> = features.to_vec().iter().map(|f| format!("{:.6}", f)).collect();
                writeln!(writer, "{},{}", feature_values.join(","), label)?;
                events_written += 1;

                // Mark as processed
                processed_approaches.insert(event_key, true);
            }
        }

        // Progress
        if idx % 50000 == 0 && idx > 0 {
            let pct = idx * 100 / candles_1m.len();
            eprintln!("  Progress: {}% ({} events)", pct, events_written);
        }
    }

    writer.flush()?;

    println!("\nExport complete in {:.1}s", process_start.elapsed().as_secs_f32());
    println!("  Total events: {}", events_written);
    println!("  Held: {} ({:.1}%)", held_count, 100.0 * held_count as f64 / events_written.max(1) as f64);
    println!("  Broke: {} ({:.1}%)", broke_count, 100.0 * broke_count as f64 / events_written.max(1) as f64);
    println!("  Output: {}", output_path);

    Ok(())
}

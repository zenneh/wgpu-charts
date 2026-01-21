//! Simple ML-based backtest.
//!
//! A minimal backtest that trades purely based on ML predictions:
//! - Long when prediction > high_threshold
//! - Short when prediction < low_threshold
//! - Exit after N candles or on opposite signal
//!
//! Usage: cargo run --bin simple_backtest --release -- --data data/btc-new.csv --model data/model_v2.onnx

use std::env;
use std::time::Instant;

use anyhow::Result;
use charter_core::Candle;
use charter_data::load_candles_from_csv;
use charter_ta::{AnalyzerConfig, MlFeatures, MlInferenceHandle, MultiTimeframeAnalyzer, ml_export_timeframes};

// ============================================================================
// Configuration
// ============================================================================

/// Threshold to go LONG (model predicts UP with high confidence)
const LONG_THRESHOLD: f32 = 0.65;

/// Threshold to go SHORT (model predicts DOWN with high confidence)
const SHORT_THRESHOLD: f32 = 0.35;

/// Hold position for this many candles max (match training lookahead!)
const HOLD_CANDLES: usize = 2;

/// Simple fee per trade (0.04% round trip - typical for maker orders)
const FEE_PCT: f32 = 0.0004;

/// Minimum candles between trades (cooldown)
const COOLDOWN: usize = 5;

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum Position {
    None,
    Long { entry_price: f32, entry_idx: usize },
    Short { entry_price: f32, entry_idx: usize },
}

struct SimpleBacktest {
    position: Position,
    balance: f32,
    initial_balance: f32,
    wins: usize,
    losses: usize,
    total_pnl: f32,
    trades: Vec<(f32, &'static str)>, // (pnl, direction)
    last_exit_idx: usize,
}

impl SimpleBacktest {
    fn new(initial_balance: f32) -> Self {
        Self {
            position: Position::None,
            balance: initial_balance,
            initial_balance,
            wins: 0,
            losses: 0,
            total_pnl: 0.0,
            trades: Vec::new(),
            last_exit_idx: 0,
        }
    }

    fn process(&mut self, candle: &Candle, prediction: f32, idx: usize) {
        let price = candle.close;

        // Check if we should exit current position
        match self.position {
            Position::Long { entry_price, entry_idx } => {
                let held = idx - entry_idx;
                // Exit on: max hold time reached
                if held >= HOLD_CANDLES {
                    let pnl_pct = (price - entry_price) / entry_price - FEE_PCT;
                    let pnl = self.balance * pnl_pct;
                    self.balance += pnl;
                    self.total_pnl += pnl;
                    self.trades.push((pnl, "LONG"));
                    if pnl > 0.0 { self.wins += 1; } else { self.losses += 1; }
                    self.position = Position::None;
                    self.last_exit_idx = idx;
                }
            }
            Position::Short { entry_price, entry_idx } => {
                let held = idx - entry_idx;
                if held >= HOLD_CANDLES {
                    let pnl_pct = (entry_price - price) / entry_price - FEE_PCT;
                    let pnl = self.balance * pnl_pct;
                    self.balance += pnl;
                    self.total_pnl += pnl;
                    self.trades.push((pnl, "SHORT"));
                    if pnl > 0.0 { self.wins += 1; } else { self.losses += 1; }
                    self.position = Position::None;
                    self.last_exit_idx = idx;
                }
            }
            Position::None => {}
        }

        // Check if we should enter a new position (with cooldown)
        if self.position == Position::None && idx >= self.last_exit_idx + COOLDOWN {
            if prediction > LONG_THRESHOLD {
                self.position = Position::Long { entry_price: price, entry_idx: idx };
            } else if prediction < SHORT_THRESHOLD {
                self.position = Position::Short { entry_price: price, entry_idx: idx };
            }
        }
    }

    fn close_position(&mut self, price: f32) {
        match self.position {
            Position::Long { entry_price, .. } => {
                let pnl_pct = (price - entry_price) / entry_price - FEE_PCT;
                let pnl = self.balance * pnl_pct;
                self.balance += pnl;
                self.total_pnl += pnl;
                self.trades.push((pnl, "LONG"));
                if pnl > 0.0 { self.wins += 1; } else { self.losses += 1; }
            }
            Position::Short { entry_price, .. } => {
                let pnl_pct = (entry_price - price) / entry_price - FEE_PCT;
                let pnl = self.balance * pnl_pct;
                self.balance += pnl;
                self.total_pnl += pnl;
                self.trades.push((pnl, "SHORT"));
                if pnl > 0.0 { self.wins += 1; } else { self.losses += 1; }
            }
            Position::None => {}
        }
        self.position = Position::None;
    }

    fn print_results(&self) {
        println!("\n══════════════════════════════════════════");
        println!("         SIMPLE BACKTEST RESULTS          ");
        println!("══════════════════════════════════════════\n");

        let total_trades = self.wins + self.losses;
        let win_rate = if total_trades > 0 {
            self.wins as f32 / total_trades as f32 * 100.0
        } else {
            0.0
        };
        let return_pct = (self.balance - self.initial_balance) / self.initial_balance * 100.0;

        println!("Initial Balance:  ${:.2}", self.initial_balance);
        println!("Final Balance:    ${:.2}", self.balance);
        println!("Total P&L:        ${:.2} ({:+.2}%)", self.total_pnl, return_pct);
        println!();
        println!("Total Trades:     {}", total_trades);
        println!("Wins:             {} ({:.1}%)", self.wins, win_rate);
        println!("Losses:           {} ({:.1}%)", self.losses, 100.0 - win_rate);
        println!();

        // Calculate profit factor
        let gross_profit: f32 = self.trades.iter().filter(|(p, _)| *p > 0.0).map(|(p, _)| *p).sum();
        let gross_loss: f32 = self.trades.iter().filter(|(p, _)| *p < 0.0).map(|(p, _)| p.abs()).sum();
        let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { f32::INFINITY };

        println!("Gross Profit:     ${:.2}", gross_profit);
        println!("Gross Loss:       ${:.2}", gross_loss);
        println!("Profit Factor:    {:.2}", profit_factor);
        println!();

        // Average win/loss
        let avg_win = if self.wins > 0 { gross_profit / self.wins as f32 } else { 0.0 };
        let avg_loss = if self.losses > 0 { gross_loss / self.losses as f32 } else { 0.0 };
        println!("Avg Win:          ${:.2}", avg_win);
        println!("Avg Loss:         ${:.2}", avg_loss);

        println!("\n══════════════════════════════════════════");
        println!("Settings: LONG > {:.0}%, SHORT < {:.0}%, Hold max {} candles",
            LONG_THRESHOLD * 100.0, SHORT_THRESHOLD * 100.0, HOLD_CANDLES);
        println!("══════════════════════════════════════════\n");
    }
}

// ============================================================================
// Feature Extraction (simplified)
// ============================================================================

fn extract_features(candle: &Candle, mta: &MultiTimeframeAnalyzer) -> Option<MlFeatures> {
    let current_price = candle.close;
    let tf_features = mta.extract_all_features(current_price);

    if tf_features.is_empty() {
        return None;
    }

    let state = mta.state();

    // Consistent distance formula
    let closest_resistance_distance = state.closest_unbroken_resistance
        .map(|(price, _)| (price - current_price) / current_price);
    let closest_support_distance = state.closest_unbroken_support
        .map(|(price, _)| (price - current_price) / current_price);

    // Price action
    let range = candle.high - candle.low;
    let (body_ratio, upper_wick_ratio, lower_wick_ratio) = if range > f32::EPSILON {
        let body = (candle.close - candle.open).abs();
        let upper = candle.high - candle.open.max(candle.close);
        let lower = candle.open.min(candle.close) - candle.low;
        (body / range, upper / range, lower / range)
    } else {
        (0.0, 0.0, 0.0)
    };
    let is_bullish = if candle.close > candle.open { 1.0 } else { 0.0 };

    Some(MlFeatures {
        timeframes: tf_features,
        current_price,
        reference_price: current_price,
        total_active_levels: state.total_active_levels() as u16,
        total_levels: state.total_levels() as u16,
        has_resistance_above: state.closest_unbroken_resistance.is_some(),
        has_support_below: state.closest_unbroken_support.is_some(),
        closest_resistance_distance,
        closest_support_distance,
        body_ratio,
        upper_wick_ratio,
        lower_wick_ratio,
        is_bullish,
        price_change_1: 0.0,
        price_change_3: 0.0,
        price_change_5: 0.0,
    })
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let start = Instant::now();

    // Parse args
    let args: Vec<String> = env::args().collect();
    let mut model_path = "data/model_v2.onnx".to_string();
    let mut data_path = "data/btc-new.csv".to_string();
    let mut max_candles = 100_000usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => { model_path = args[i + 1].clone(); i += 2; }
            "--data" if i + 1 < args.len() => { data_path = args[i + 1].clone(); i += 2; }
            "--candles" if i + 1 < args.len() => { max_candles = args[i + 1].parse().unwrap_or(100_000); i += 2; }
            _ => i += 1,
        }
    }

    println!("Simple ML Backtest");
    println!("==================\n");

    // Load model
    print!("Loading model... ");
    let ml = MlInferenceHandle::load(&model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
    println!("OK");

    // Load data
    print!("Loading data... ");
    let all_candles = load_candles_from_csv(&data_path)?;
    let candles: Vec<Candle> = if all_candles.len() > max_candles {
        all_candles[all_candles.len() - max_candles..].to_vec()
    } else {
        all_candles
    };
    println!("{} candles", candles.len());

    // Setup analyzer
    let timeframes = ml_export_timeframes();
    let config = AnalyzerConfig::default();
    let mut mta = MultiTimeframeAnalyzer::with_timeframes(timeframes, config);

    // Setup backtest
    let mut backtest = SimpleBacktest::new(1000.0);
    let warmup = 2000;

    println!("Running backtest on {} candles...\n", candles.len());

    let mut predictions_made = 0;
    let mut prediction_sum = 0.0;
    let mut pred_min = 1.0f32;
    let mut pred_max = 0.0f32;
    let mut above_60 = 0;
    let mut below_40 = 0;

    for (idx, candle) in candles.iter().enumerate() {
        mta.process_1m_candle(candle);

        if idx < warmup {
            continue;
        }

        if let Some(features) = extract_features(candle, &mta) {
            if features.timeframes.len() >= 5 {
                if let Ok(pred) = ml.predict(&features) {
                    let p = pred.direction_up_prob;
                    prediction_sum += p;
                    predictions_made += 1;
                    pred_min = pred_min.min(p);
                    pred_max = pred_max.max(p);
                    if p > 0.60 { above_60 += 1; }
                    if p < 0.40 { below_40 += 1; }
                    backtest.process(candle, p, idx);
                }
            }
        }

        // Progress
        if idx % 20000 == 0 && idx > 0 {
            print!("\r  Progress: {}%", idx * 100 / candles.len());
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }

    // Close any open position
    if let Some(last) = candles.last() {
        backtest.close_position(last.close);
    }

    println!("\r  Progress: 100%\n");

    // Stats
    let avg_pred = if predictions_made > 0 { prediction_sum / predictions_made as f32 } else { 0.5 };
    println!("Predictions made: {}", predictions_made);
    println!("Avg prediction:   {:.3}", avg_pred);
    println!("Prediction range: {:.3} - {:.3}", pred_min, pred_max);
    println!("Above 60%: {} ({:.1}%)", above_60, 100.0 * above_60 as f32 / predictions_made as f32);
    println!("Below 40%: {} ({:.1}%)", below_40, 100.0 * below_40 as f32 / predictions_made as f32);

    backtest.print_results();

    println!("Completed in {:.1}s", start.elapsed().as_secs_f32());

    Ok(())
}

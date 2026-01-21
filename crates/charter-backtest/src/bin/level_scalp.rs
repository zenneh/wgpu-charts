//! Level Scalping Backtest with Leveraged Trading
//!
//! Simulates actual leveraged trades with proper entry, stop loss, and take profit.
//!
//! Usage: cargo run -p charter-backtest --bin level_scalp --release -- \
//!     --data data/btc-new.csv --model data/level_model.json --scaler data/level_model.scaler.json

use std::env;
use std::fs;
use std::time::Instant;

use anyhow::{Context, Result};
use charter_core::Candle;
use charter_data::load_candles_from_csv;
use charter_ta::{
    extract_level_features, is_approaching_level,
    Analyzer, AnalyzerConfig, DefaultAnalyzer, Level, LevelDirection, LevelEventFeatures,
    TimeframeConfig, BREAK_THRESHOLD, HOLD_THRESHOLD,
};
use charter_core::Timeframe;
use serde::Deserialize;
use xgboost_rust::Booster;

// ============================================================================
// Configuration
// ============================================================================

/// Confidence threshold to take a BOUNCE trade (predict level will HOLD)
const HOLD_CONFIDENCE: f32 = 0.65;

/// Confidence threshold to take a BREAKOUT trade (predict level will BREAK)
const BREAK_CONFIDENCE: f32 = 0.35;

/// Leverage multiplier
const LEVERAGE: f32 = 10.0;

/// Risk per trade as fraction of balance
const RISK_PER_TRADE: f32 = 0.02; // 2% risk per trade

/// Trading fee per side (entry + exit)
const FEE_PER_SIDE: f32 = 0.0004; // 0.04% per side = 0.08% round trip

/// Maximum candles to hold a position before timeout
const MAX_HOLD_CANDLES: usize = 20;

/// Cooldown candles between trades
const COOLDOWN: usize = 5;

/// TP/SL distances from entry (not from level)
const TP_DISTANCE: f32 = 0.004; // 0.4% take profit from entry
const SL_DISTANCE: f32 = 0.002; // 0.2% stop loss from entry (2:1 R:R ratio)

// ============================================================================
// Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
enum TradeType {
    Bounce,
    Breakout,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TradeDirection {
    Long,
    Short,
}

#[derive(Debug, Clone)]
struct Position {
    trade_type: TradeType,
    direction: TradeDirection,
    entry_price: f32,
    stop_loss: f32,
    take_profit: f32,
    position_size: f32, // In units of the asset
    entry_idx: usize,
    level_price: f32,
    prediction: f32,
}

#[derive(Debug)]
struct TradeResult {
    trade_type: TradeType,
    direction: TradeDirection,
    entry_price: f32,
    exit_price: f32,
    pnl: f32,
    pnl_pct: f32,
    exit_reason: &'static str,
    hold_candles: usize,
}

struct LevelScalpBacktest {
    balance: f32,
    initial_balance: f32,
    position: Option<Position>,
    trades: Vec<TradeResult>,
    last_trade_idx: usize,
    // Prediction stats
    all_predictions: Vec<f32>,
    bounce_predictions: Vec<f32>,
    breakout_predictions: Vec<f32>,
}

impl LevelScalpBacktest {
    fn new(initial_balance: f32) -> Self {
        Self {
            balance: initial_balance,
            initial_balance,
            position: None,
            trades: Vec::new(),
            last_trade_idx: 0,
            all_predictions: Vec::new(),
            bounce_predictions: Vec::new(),
            breakout_predictions: Vec::new(),
        }
    }

    fn can_trade(&self, idx: usize) -> bool {
        self.position.is_none() && idx > self.last_trade_idx + COOLDOWN
    }

    /// Enter a bounce trade (expecting level to hold)
    fn enter_bounce(&mut self, level: &Level, current_price: f32, prediction: f32, idx: usize) {
        // SL at level + buffer (invalidates bounce thesis)
        // TP at 1.5x the risk distance for positive expectancy with >40% win rate
        let (direction, tp, sl, sl_distance) = match level.level_direction {
            LevelDirection::Support => {
                // Support hold = price bounces UP = go LONG
                // SL just below the level (if price breaks below, we're wrong)
                let sl = level.price * (1.0 - SL_DISTANCE);
                let distance_to_sl = (current_price - sl) / current_price;
                let tp = current_price * (1.0 + distance_to_sl * 1.5); // 1.5:1 R:R
                (TradeDirection::Long, tp, sl, distance_to_sl)
            }
            LevelDirection::Resistance => {
                // Resistance hold = price bounces DOWN = go SHORT
                // SL just above the level
                let sl = level.price * (1.0 + SL_DISTANCE);
                let distance_to_sl = (sl - current_price) / current_price;
                let tp = current_price * (1.0 - distance_to_sl * 1.5); // 1.5:1 R:R
                (TradeDirection::Short, tp, sl, distance_to_sl)
            }
        };

        // Calculate position size based on actual SL distance
        let risk_amount = self.balance * RISK_PER_TRADE;
        let loss_per_unit = sl_distance * LEVERAGE + FEE_PER_SIDE * 2.0 * LEVERAGE;
        let position_value = risk_amount / loss_per_unit;
        let position_size = position_value / current_price;

        self.position = Some(Position {
            trade_type: TradeType::Bounce,
            direction,
            entry_price: current_price,
            stop_loss: sl,
            take_profit: tp,
            position_size,
            entry_idx: idx,
            level_price: level.price,
            prediction,
        });

        self.all_predictions.push(prediction);
        self.bounce_predictions.push(prediction);
    }

    /// Enter a breakout trade (expecting level to break)
    fn enter_breakout(&mut self, level: &Level, current_price: f32, prediction: f32, idx: usize) {
        // For breakout: enter in direction of expected break
        // SL at opposite side (if level holds, we're wrong)
        // TP at level + continuation distance
        let (direction, tp, sl, sl_distance) = match level.level_direction {
            LevelDirection::Support => {
                // Support break = price goes DOWN = go SHORT
                // SL above entry (if price bounces, we're wrong)
                let sl = current_price * (1.0 + SL_DISTANCE);
                let distance_to_sl = (sl - current_price) / current_price;
                // TP at level price minus same distance (continuation through level)
                let tp = level.price * (1.0 - distance_to_sl * 1.5);
                (TradeDirection::Short, tp, sl, distance_to_sl)
            }
            LevelDirection::Resistance => {
                // Resistance break = price goes UP = go LONG
                // SL below entry
                let sl = current_price * (1.0 - SL_DISTANCE);
                let distance_to_sl = (current_price - sl) / current_price;
                // TP at level price plus same distance
                let tp = level.price * (1.0 + distance_to_sl * 1.5);
                (TradeDirection::Long, tp, sl, distance_to_sl)
            }
        };

        // Calculate position size based on actual SL distance
        let risk_amount = self.balance * RISK_PER_TRADE;
        let loss_per_unit = sl_distance * LEVERAGE + FEE_PER_SIDE * 2.0 * LEVERAGE;
        let position_value = risk_amount / loss_per_unit;
        let position_size = position_value / current_price;

        self.position = Some(Position {
            trade_type: TradeType::Breakout,
            direction,
            entry_price: current_price,
            stop_loss: sl,
            take_profit: tp,
            position_size,
            entry_idx: idx,
            level_price: level.price,
            prediction,
        });

        self.all_predictions.push(prediction);
        self.breakout_predictions.push(prediction);
    }

    /// Check if position should be closed based on candle price action
    fn check_exit(&mut self, candle: &Candle, idx: usize) -> bool {
        let position = match &self.position {
            Some(p) => p.clone(),
            None => return false,
        };

        let hold_candles = idx - position.entry_idx;

        // Determine exit price and reason
        let (exit_price, reason) = match position.direction {
            TradeDirection::Long => {
                // Check SL first (more conservative)
                if candle.low <= position.stop_loss {
                    (position.stop_loss, "SL")
                } else if candle.high >= position.take_profit {
                    (position.take_profit, "TP")
                } else if hold_candles >= MAX_HOLD_CANDLES {
                    (candle.close, "TIMEOUT")
                } else {
                    return false;
                }
            }
            TradeDirection::Short => {
                // Check SL first (more conservative)
                if candle.high >= position.stop_loss {
                    (position.stop_loss, "SL")
                } else if candle.low <= position.take_profit {
                    (position.take_profit, "TP")
                } else if hold_candles >= MAX_HOLD_CANDLES {
                    (candle.close, "TIMEOUT")
                } else {
                    return false;
                }
            }
        };

        // Calculate P&L
        let price_change = match position.direction {
            TradeDirection::Long => (exit_price - position.entry_price) / position.entry_price,
            TradeDirection::Short => (position.entry_price - exit_price) / position.entry_price,
        };

        // Apply leverage and fees
        let leveraged_return = price_change * LEVERAGE;
        let fees = FEE_PER_SIDE * 2.0 * LEVERAGE; // Entry + exit fees, scaled by leverage
        let net_return = leveraged_return - fees;

        // P&L on the risked amount
        let position_value = position.position_size * position.entry_price;
        let pnl = position_value * net_return;

        // Update balance
        self.balance += pnl;
        self.last_trade_idx = idx;

        self.trades.push(TradeResult {
            trade_type: position.trade_type,
            direction: position.direction,
            entry_price: position.entry_price,
            exit_price,
            pnl,
            pnl_pct: net_return * 100.0,
            exit_reason: reason,
            hold_candles,
        });

        self.position = None;
        true
    }

    fn print_prediction_stats(&self) {
        if self.all_predictions.is_empty() {
            return;
        }

        let min = self.all_predictions.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self.all_predictions.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = self.all_predictions.iter().sum::<f32>() / self.all_predictions.len() as f32;

        println!("\nPrediction Distribution:");
        println!("  Total signals: {}", self.all_predictions.len());
        println!("  Min: {:.3}, Max: {:.3}, Mean: {:.3}", min, max, mean);
        println!("  Bounce signals: {} (avg prob: {:.3})",
            self.bounce_predictions.len(),
            if self.bounce_predictions.is_empty() { 0.0 }
            else { self.bounce_predictions.iter().sum::<f32>() / self.bounce_predictions.len() as f32 }
        );
        println!("  Breakout signals: {} (avg prob: {:.3})",
            self.breakout_predictions.len(),
            if self.breakout_predictions.is_empty() { 0.0 }
            else { self.breakout_predictions.iter().sum::<f32>() / self.breakout_predictions.len() as f32 }
        );
    }

    fn print_results(&self) {
        println!("\n══════════════════════════════════════════════════════");
        println!("         LEVERAGED LEVEL SCALP BACKTEST RESULTS       ");
        println!("══════════════════════════════════════════════════════\n");

        // Overall stats
        let total = self.trades.len();
        let wins: usize = self.trades.iter().filter(|t| t.pnl > 0.0).count();
        let losses = total - wins;
        let win_rate = if total > 0 { 100.0 * wins as f32 / total as f32 } else { 0.0 };
        let return_pct = 100.0 * (self.balance - self.initial_balance) / self.initial_balance;
        let total_pnl: f32 = self.trades.iter().map(|t| t.pnl).sum();

        println!("Initial Balance: ${:.2}", self.initial_balance);
        println!("Final Balance:   ${:.2}", self.balance);
        println!("Total P&L:       ${:.2} ({:+.2}%)", total_pnl, return_pct);
        println!();
        println!("Total Trades:    {}", total);
        println!("Wins:            {} ({:.1}%)", wins, win_rate);
        println!("Losses:          {} ({:.1}%)", losses, 100.0 - win_rate);
        println!();

        // Exit reasons
        let tp_count = self.trades.iter().filter(|t| t.exit_reason == "TP").count();
        let sl_count = self.trades.iter().filter(|t| t.exit_reason == "SL").count();
        let to_count = self.trades.iter().filter(|t| t.exit_reason == "TIMEOUT").count();

        println!("Exit Reasons:");
        println!("  Take Profit:   {}", tp_count);
        println!("  Stop Loss:     {}", sl_count);
        println!("  Timeout:       {}", to_count);
        println!();

        // Breakdown by trade type
        let bounce_trades: Vec<_> = self.trades.iter().filter(|t| t.trade_type == TradeType::Bounce).collect();
        let breakout_trades: Vec<_> = self.trades.iter().filter(|t| t.trade_type == TradeType::Breakout).collect();

        let bounce_wins = bounce_trades.iter().filter(|t| t.pnl > 0.0).count();
        let breakout_wins = breakout_trades.iter().filter(|t| t.pnl > 0.0).count();

        let bounce_pnl: f32 = bounce_trades.iter().map(|t| t.pnl).sum();
        let breakout_pnl: f32 = breakout_trades.iter().map(|t| t.pnl).sum();

        println!("By Trade Type:");
        println!("  Bounce:   {} trades, {:.1}% win rate, ${:.2} P&L",
            bounce_trades.len(),
            if bounce_trades.is_empty() { 0.0 } else { 100.0 * bounce_wins as f32 / bounce_trades.len() as f32 },
            bounce_pnl
        );
        println!("  Breakout: {} trades, {:.1}% win rate, ${:.2} P&L",
            breakout_trades.len(),
            if breakout_trades.is_empty() { 0.0 } else { 100.0 * breakout_wins as f32 / breakout_trades.len() as f32 },
            breakout_pnl
        );
        println!();

        // By direction
        let long_trades: Vec<_> = self.trades.iter().filter(|t| t.direction == TradeDirection::Long).collect();
        let short_trades: Vec<_> = self.trades.iter().filter(|t| t.direction == TradeDirection::Short).collect();

        let long_wins = long_trades.iter().filter(|t| t.pnl > 0.0).count();
        let short_wins = short_trades.iter().filter(|t| t.pnl > 0.0).count();

        println!("By Direction:");
        println!("  Long:  {} trades, {:.1}% win rate",
            long_trades.len(),
            if long_trades.is_empty() { 0.0 } else { 100.0 * long_wins as f32 / long_trades.len() as f32 }
        );
        println!("  Short: {} trades, {:.1}% win rate",
            short_trades.len(),
            if short_trades.is_empty() { 0.0 } else { 100.0 * short_wins as f32 / short_trades.len() as f32 }
        );
        println!();

        // Profit factor
        let gross_profit: f32 = self.trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_loss: f32 = self.trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
        let pf = if gross_loss > 0.0 { gross_profit / gross_loss } else { f32::INFINITY };

        println!("Gross Profit:    ${:.2}", gross_profit);
        println!("Gross Loss:      ${:.2}", gross_loss);
        println!("Profit Factor:   {:.2}", pf);

        // Average trade stats
        if !self.trades.is_empty() {
            let avg_win: f32 = {
                let winning: Vec<_> = self.trades.iter().filter(|t| t.pnl > 0.0).collect();
                if winning.is_empty() { 0.0 }
                else { winning.iter().map(|t| t.pnl).sum::<f32>() / winning.len() as f32 }
            };
            let avg_loss: f32 = {
                let losing: Vec<_> = self.trades.iter().filter(|t| t.pnl < 0.0).collect();
                if losing.is_empty() { 0.0 }
                else { losing.iter().map(|t| t.pnl.abs()).sum::<f32>() / losing.len() as f32 }
            };
            let avg_hold: f32 = self.trades.iter().map(|t| t.hold_candles as f32).sum::<f32>() / self.trades.len() as f32;

            println!();
            println!("Avg Win:         ${:.2}", avg_win);
            println!("Avg Loss:        ${:.2}", avg_loss);
            println!("Avg Hold:        {:.1} candles", avg_hold);
        }

        println!("\n══════════════════════════════════════════════════════");
        println!("Settings: {}x leverage, {:.1}% risk/trade, {:.2}% fee/side",
            LEVERAGE, RISK_PER_TRADE * 100.0, FEE_PER_SIDE * 100.0);
        println!("Thresholds: Hold >= {:.0}% | Break <= {:.0}%",
            HOLD_CONFIDENCE * 100.0, BREAK_CONFIDENCE * 100.0);
        println!("TP/SL: {:.2}% / {:.2}% (from entry)",
            TP_DISTANCE * 100.0, SL_DISTANCE * 100.0);
        println!("Max hold: {} candles", MAX_HOLD_CANDLES);
        println!("══════════════════════════════════════════════════════\n");
    }
}

// ============================================================================
// ML Model Inference
// ============================================================================

/// Scaler data loaded from JSON
#[derive(Debug, Deserialize)]
struct ScalerData {
    mean: Vec<f64>,
    scale: Vec<f64>,
}

/// ML predictor that uses trained XGBoost model
struct LevelPredictor {
    model: Booster,
    scaler: ScalerData,
    num_features: usize,
}

impl LevelPredictor {
    fn load(model_path: &str, scaler_path: &str) -> Result<Self> {
        let model = Booster::load(model_path)
            .context("Failed to load XGBoost model")?;

        let scaler_json = fs::read_to_string(scaler_path)
            .context("Failed to read scaler file")?;
        let scaler: ScalerData = serde_json::from_str(&scaler_json)
            .context("Failed to parse scaler JSON")?;

        let num_features = scaler.mean.len();

        Ok(Self { model, scaler, num_features })
    }

    fn scale_features(&self, features: &[f32]) -> Vec<f32> {
        features.iter()
            .zip(self.scaler.mean.iter().zip(self.scaler.scale.iter()))
            .map(|(&f, (&mean, &scale))| {
                ((f as f64 - mean) / scale) as f32
            })
            .collect()
    }

    fn predict(&self, features: &LevelEventFeatures) -> f32 {
        let raw_features = features.to_vec();
        let scaled = self.scale_features(&raw_features);

        let predictions = self.model.predict(&scaled, 1, self.num_features, 0, false);

        match predictions {
            Ok(preds) if !preds.is_empty() => preds[0],
            _ => 0.5,
        }
    }
}

/// Fallback heuristic when model is not available
fn predict_hold_probability_heuristic(features: &LevelEventFeatures) -> f32 {
    let mut score = 0.5;

    score += (features.respected_ratio - 0.5) * 0.3;

    let hits = features.prior_hit_count as f32;
    if hits >= 3.0 && features.respected_ratio > 0.6 {
        score += 0.1;
    }

    score += features.level_timeframe_strength * 0.1;

    let is_support = features.is_support > 0.5;
    if is_support && features.current_lower_wick > 0.3 {
        score += 0.1;
    }
    if !is_support && features.current_upper_wick > 0.3 {
        score += 0.1;
    }

    score.clamp(0.0, 1.0)
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let start = Instant::now();

    // Parse args
    let args: Vec<String> = env::args().collect();
    let mut data_path = "data/btc-new.csv".to_string();
    let mut model_path: Option<String> = None;
    let mut scaler_path: Option<String> = None;
    let mut max_candles = 100_000usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data" if i + 1 < args.len() => { data_path = args[i + 1].clone(); i += 2; }
            "--model" if i + 1 < args.len() => { model_path = Some(args[i + 1].clone()); i += 2; }
            "--scaler" if i + 1 < args.len() => { scaler_path = Some(args[i + 1].clone()); i += 2; }
            "--candles" if i + 1 < args.len() => { max_candles = args[i + 1].parse().unwrap_or(100_000); i += 2; }
            _ => i += 1,
        }
    }

    println!("Leveraged Level Scalp Backtest");
    println!("==============================\n");

    // Load ML model if paths provided
    let predictor: Option<LevelPredictor> = match (model_path, scaler_path) {
        (Some(m), Some(s)) => {
            print!("Loading model... ");
            match LevelPredictor::load(&m, &s) {
                Ok(p) => {
                    println!("OK ({} features)", p.num_features);
                    Some(p)
                }
                Err(e) => {
                    println!("FAILED: {}", e);
                    println!("Falling back to heuristic predictor\n");
                    None
                }
            }
        }
        _ => {
            println!("No model specified, using heuristic predictor\n");
            None
        }
    };

    // Load data
    print!("Loading data... ");
    let all_candles = load_candles_from_csv(&data_path)?;
    let candles: Vec<Candle> = if all_candles.len() > max_candles {
        all_candles[all_candles.len() - max_candles..].to_vec()
    } else {
        all_candles
    };
    println!("{} candles", candles.len());

    // Setup analyzer with multiple timeframes
    let timeframes = vec![
        Timeframe::Min1,
        Timeframe::Min3,
        Timeframe::Min5,
        Timeframe::Min30,
        Timeframe::Hour1,
        Timeframe::Day1,
        Timeframe::Week1,
    ];

    let tf_configs: Vec<TimeframeConfig> = timeframes
        .iter()
        .map(|&tf| TimeframeConfig::new(tf, 3, 0.001))
        .collect();
    let config = AnalyzerConfig::new(tf_configs.clone());
    let mut analyzer = DefaultAnalyzer::new(config);

    // Pre-aggregate candles
    let mut tf_candles: Vec<Vec<Candle>> = Vec::new();
    for &tf in &timeframes {
        let agg = if tf == Timeframe::Min1 {
            candles.clone()
        } else {
            charter_ta::aggregate_candles(&candles, tf)
        };
        tf_candles.push(agg);
    }

    // Pre-compute timeframe candle counts
    let mut tf_counts: Vec<Vec<usize>> = Vec::new();
    for (_tf_idx, &tf) in timeframes.iter().enumerate() {
        let mut counts = Vec::with_capacity(candles.len());
        if tf == Timeframe::Min1 {
            for i in 0..candles.len() {
                counts.push(i + 1);
            }
        } else {
            let mut count = 0;
            let mut last_bucket: Option<i64> = None;
            for candle in &candles {
                let bucket = (candle.timestamp / tf.seconds()).floor() as i64;
                if last_bucket.map_or(true, |lb| bucket != lb) {
                    if last_bucket.is_some() {
                        count += 1;
                    }
                    last_bucket = Some(bucket);
                }
                counts.push(count);
            }
        }
        tf_counts.push(counts);
    }

    // Setup backtest
    let mut backtest = LevelScalpBacktest::new(1000.0);
    let warmup = 2000;

    println!("Running backtest...\n");

    let mut last_tf_counts: Vec<usize> = vec![0; timeframes.len()];
    let mut signals_found = 0;

    for idx in 0..candles.len() {
        let candle = &candles[idx];
        let current_price = candle.close;

        // Update analyzer
        for (tf_idx, _tf) in timeframes.iter().enumerate() {
            let tf_candle_count = tf_counts[tf_idx][idx];
            if tf_candle_count > last_tf_counts[tf_idx] && tf_candle_count > 0 {
                let tf_candle_slice = &tf_candles[tf_idx][..tf_candle_count];
                analyzer.update(tf_idx as u8, tf_candle_slice, current_price);
                last_tf_counts[tf_idx] = tf_candle_count;
            }
        }

        if idx < warmup {
            continue;
        }

        // Check exits first
        backtest.check_exit(candle, idx);

        // Look for entry signals
        if !backtest.can_trade(idx) {
            continue;
        }

        // Get recent candles for momentum
        let momentum_start = idx.saturating_sub(5);
        let recent_candles = &candles[momentum_start..idx];
        if recent_candles.len() < 3 {
            continue;
        }

        // Check all active levels for approach
        let state = analyzer.state();

        // Track best bounce and breakout signals
        let mut best_bounce: Option<(f32, Level)> = None;
        let mut best_breakout: Option<(f32, Level)> = None;

        for (_tf_idx, tf_state) in &state.timeframe_states {
            for level in tf_state.level_index.active_levels() {
                if !is_approaching_level(level, current_price) {
                    continue;
                }

                // Count nearby levels
                let nearby = tf_state.level_index
                    .levels_in_range(current_price * 0.99, current_price * 1.01)
                    .len();

                // Get opposite level distance
                let opposite_distance = match level.level_direction {
                    LevelDirection::Support => {
                        tf_state.level_index.closest_resistance_above(current_price)
                            .map(|l| (l.price - current_price) / current_price)
                    }
                    LevelDirection::Resistance => {
                        tf_state.level_index.closest_support_below(current_price)
                            .map(|l| (current_price - l.price) / current_price)
                    }
                };

                let features = extract_level_features(
                    level,
                    current_price,
                    candle,
                    recent_candles,
                    idx,
                    nearby,
                    opposite_distance,
                );

                let hold_prob = match &predictor {
                    Some(p) => p.predict(&features),
                    None => predict_hold_probability_heuristic(&features),
                };

                // Check for bounce signal (level will hold)
                if hold_prob >= HOLD_CONFIDENCE {
                    if best_bounce.as_ref().map_or(true, |(best_prob, _)| hold_prob > *best_prob) {
                        best_bounce = Some((hold_prob, level.clone()));
                    }
                }

                // Check for breakout signal (level will break)
                if hold_prob <= BREAK_CONFIDENCE {
                    if best_breakout.as_ref().map_or(true, |(best_prob, _)| hold_prob < *best_prob) {
                        best_breakout = Some((hold_prob, level.clone()));
                    }
                }
            }
        }

        // Execute best signal (prefer bounce over breakout)
        if let Some((prob, level)) = best_bounce {
            signals_found += 1;
            backtest.enter_bounce(&level, current_price, prob, idx);
        } else if let Some((prob, level)) = best_breakout {
            signals_found += 1;
            backtest.enter_breakout(&level, current_price, prob, idx);
        }

        // Progress
        if idx % 20000 == 0 && idx > 0 {
            print!("\r  Progress: {}%", idx * 100 / candles.len());
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }

    // Close any open position at market
    if let Some(last_candle) = candles.last() {
        backtest.check_exit(last_candle, candles.len());
    }

    println!("\r  Progress: 100%\n");
    println!("Signals found: {}", signals_found);

    backtest.print_results();
    backtest.print_prediction_stats();

    println!("\nCompleted in {:.1}s", start.elapsed().as_secs_f32());

    Ok(())
}

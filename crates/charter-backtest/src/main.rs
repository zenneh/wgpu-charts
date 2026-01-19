//! Backtest trading strategy using ML predictions.
//!
//! This module runs a realistic backtest with:
//! - ML model predictions for direction
//! - Position sizing based on confidence
//! - Stop loss and take profit
//! - Bankroll management
//!
//! Usage: charter-backtest [--model PATH] [--data PATH] [--initial-balance N]

use std::env;
use std::time::Instant;

use anyhow::Result;
use charter_core::{aggregate_candles, Candle, Timeframe};
use charter_data::load_candles_from_csv;
use charter_ta::{
    Analyzer, AnalyzerConfig, MlFeatures, MlInferenceHandle, TimeframeFeatures,
};

// ============================================================================
// Configuration
// ============================================================================

/// Minimum confidence threshold to enter a trade (0.0 - 1.0).
const MIN_CONFIDENCE: f32 = 0.15;

/// Risk per trade as fraction of bankroll.
const RISK_PER_TRADE: f32 = 0.02; // 2%

/// Stop loss as percentage of entry price.
const STOP_LOSS_PCT: f32 = 0.005; // 0.5%

/// Take profit as percentage of entry price.
const TAKE_PROFIT_PCT: f32 = 0.01; // 1.0%

/// Trading fee per trade (taker fee).
const FEE_PCT: f32 = 0.0001; // 0.01%

/// Leverage multiplier (1x = no leverage, 100x = max).
const LEVERAGE: f32 = 1.0;

/// Minimum candles for TA warmup.
const MIN_CANDLES: usize = 100;

/// Timeframe indices for ML features: 5m(2), 1h(4), 1d(8), 1w(9)
const ML_TIMEFRAME_INDICES: [usize; 4] = [2, 4, 8, 9];

// ============================================================================
// Trading Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
enum Direction {
    Long,
    Short,
}

#[derive(Debug, Clone, serde::Serialize)]
struct Trade {
    direction: Direction,
    entry_price: f32,
    entry_time: f64,
    margin: f32,         // Margin (collateral) in USD
    size: f32,           // Position size in USD (margin * leverage)
    leverage: f32,
    stop_loss: f32,
    take_profit: f32,
    liquidation_price: f32,
    exit_price: Option<f32>,
    exit_time: Option<f64>,
    pnl: Option<f32>,
    exit_reason: Option<String>,
}

impl Trade {
    fn new(direction: Direction, entry_price: f32, entry_time: f64, margin: f32, leverage: f32) -> Self {
        let size = margin * leverage;

        // Stop loss and take profit are tighter with leverage
        // At high leverage, we need to be more conservative
        let sl_pct = STOP_LOSS_PCT;
        let tp_pct = TAKE_PROFIT_PCT;

        let (stop_loss, take_profit, liquidation_price) = match direction {
            Direction::Long => (
                entry_price * (1.0 - sl_pct),
                entry_price * (1.0 + tp_pct),
                // Liquidation when loss >= margin (e.g., at 10x, 10% drop = liquidation)
                entry_price * (1.0 - 0.99 / leverage),
            ),
            Direction::Short => (
                entry_price * (1.0 + sl_pct),
                entry_price * (1.0 - tp_pct),
                entry_price * (1.0 + 0.99 / leverage),
            ),
        };

        Self {
            direction,
            entry_price,
            entry_time,
            margin,
            size,
            leverage,
            stop_loss,
            take_profit,
            liquidation_price,
            exit_price: None,
            exit_time: None,
            pnl: None,
            exit_reason: None,
        }
    }

    fn check_exit(&mut self, candle: &Candle) -> bool {
        let (hit_sl, hit_tp, hit_liq) = match self.direction {
            Direction::Long => (
                candle.low <= self.stop_loss,
                candle.high >= self.take_profit,
                candle.low <= self.liquidation_price,
            ),
            Direction::Short => (
                candle.high >= self.stop_loss,
                candle.low <= self.take_profit,
                candle.high >= self.liquidation_price,
            ),
        };

        // Liquidation takes priority (worst case)
        if hit_liq {
            self.close(self.liquidation_price, candle.timestamp, "liquidation");
            return true;
        }

        if hit_sl {
            self.close(self.stop_loss, candle.timestamp, "stop_loss");
            return true;
        }

        if hit_tp {
            self.close(self.take_profit, candle.timestamp, "take_profit");
            return true;
        }

        false
    }

    fn close(&mut self, exit_price: f32, exit_time: f64, reason: &str) {
        self.exit_price = Some(exit_price);
        self.exit_time = Some(exit_time);
        self.exit_reason = Some(reason.to_string());

        // Calculate PnL with leverage
        let price_change_pct = match self.direction {
            Direction::Long => (exit_price - self.entry_price) / self.entry_price,
            Direction::Short => (self.entry_price - exit_price) / self.entry_price,
        };

        // PnL = margin * leverage * price_change - fees
        // Fees are based on position size (margin * leverage)
        let fees = self.size * FEE_PCT * 2.0;
        let raw_pnl = self.margin * self.leverage * price_change_pct - fees;

        // On liquidation, lose entire margin
        if reason == "liquidation" {
            self.pnl = Some(-self.margin);
        } else {
            self.pnl = Some(raw_pnl);
        }
    }
}

// ============================================================================
// Backtest Engine
// ============================================================================

struct BacktestEngine {
    initial_balance: f32,
    balance: f32,
    trades: Vec<Trade>,
    current_trade: Option<Trade>,
    ml_inference: MlInferenceHandle,

    // Stats
    total_predictions: usize,
    correct_predictions: usize,
}

impl BacktestEngine {
    fn new(initial_balance: f32, ml_inference: MlInferenceHandle) -> Self {
        Self {
            initial_balance,
            balance: initial_balance,
            trades: Vec::new(),
            current_trade: None,
            ml_inference,
            total_predictions: 0,
            correct_predictions: 0,
        }
    }

    fn process_candle(
        &mut self,
        candle: &Candle,
        features: Option<&MlFeatures>,
        leverage: f32,
    ) -> Result<()> {
        // Check if current trade should be closed
        if let Some(ref mut trade) = self.current_trade {
            if trade.check_exit(candle) {
                let pnl = trade.pnl.unwrap_or(0.0);
                self.balance += pnl;
                self.trades.push(trade.clone());
                self.current_trade = None;
            }
        }

        // Try to open new trade if no position
        if self.current_trade.is_none() {
            if let Some(features) = features {
                self.try_open_trade(candle, features, leverage)?;
            }
        }

        Ok(())
    }

    fn try_open_trade(&mut self, candle: &Candle, features: &MlFeatures, leverage: f32) -> Result<()> {
        // Get ML prediction
        let prediction = match self.ml_inference.predict(features) {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        self.total_predictions += 1;

        // Check confidence threshold
        if prediction.confidence < MIN_CONFIDENCE {
            return Ok(());
        }

        // Determine direction
        let direction = if prediction.direction_up_prob > 0.5 {
            Direction::Long
        } else {
            Direction::Short
        };

        // Calculate margin (collateral) based on risk
        // With leverage, we risk more per unit of margin
        // Margin = (Balance * Risk%) / (StopLoss% * Leverage)
        let margin = (self.balance * RISK_PER_TRADE) / (STOP_LOSS_PCT * leverage);
        let margin = margin.min(self.balance * 0.5); // Max 50% of balance as margin

        if margin < 10.0 {
            return Ok(()); // Min margin
        }

        // Open trade with leverage
        let trade = Trade::new(direction, candle.close, candle.timestamp, margin, leverage);
        self.current_trade = Some(trade);

        Ok(())
    }

    fn close_open_trade(&mut self, candle: &Candle) {
        if let Some(ref mut trade) = self.current_trade {
            trade.close(candle.close, candle.timestamp, "end_of_backtest");
            let pnl = trade.pnl.unwrap_or(0.0);
            self.balance += pnl;
            self.trades.push(trade.clone());
            self.current_trade = None;
        }
    }

    fn print_results(&self) {
        println!();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║                   BACKTEST RESULTS                           ║");
        println!("╚══════════════════════════════════════════════════════════════╝");
        println!();

        // Basic stats
        let total_trades = self.trades.len();
        let winning_trades: Vec<_> = self.trades.iter().filter(|t| t.pnl.unwrap_or(0.0) > 0.0).collect();
        let losing_trades: Vec<_> = self.trades.iter().filter(|t| t.pnl.unwrap_or(0.0) <= 0.0).collect();

        let total_pnl: f32 = self.trades.iter().map(|t| t.pnl.unwrap_or(0.0)).sum();
        let gross_profit: f32 = winning_trades.iter().map(|t| t.pnl.unwrap_or(0.0)).sum();
        let gross_loss: f32 = losing_trades.iter().map(|t| t.pnl.unwrap_or(0.0)).sum();

        let win_rate = if total_trades > 0 {
            winning_trades.len() as f32 / total_trades as f32 * 100.0
        } else {
            0.0
        };

        let avg_win = if !winning_trades.is_empty() {
            gross_profit / winning_trades.len() as f32
        } else {
            0.0
        };

        let avg_loss = if !losing_trades.is_empty() {
            gross_loss / losing_trades.len() as f32
        } else {
            0.0
        };

        let profit_factor = if gross_loss.abs() > 0.0 {
            gross_profit / gross_loss.abs()
        } else {
            f32::INFINITY
        };

        let return_pct = (self.balance - self.initial_balance) / self.initial_balance * 100.0;

        // Exit reason breakdown
        let sl_exits = self.trades.iter().filter(|t| t.exit_reason.as_deref() == Some("stop_loss")).count();
        let tp_exits = self.trades.iter().filter(|t| t.exit_reason.as_deref() == Some("take_profit")).count();

        println!("📊 PERFORMANCE SUMMARY");
        println!("   ─────────────────────────────────────");
        println!("   Initial Balance:    ${:>12.2}", self.initial_balance);
        println!("   Final Balance:      ${:>12.2}", self.balance);
        println!("   Total P&L:          ${:>12.2} ({:+.2}%)", total_pnl, return_pct);
        println!();

        println!("📈 TRADE STATISTICS");
        println!("   ─────────────────────────────────────");
        println!("   Total Trades:       {:>12}", total_trades);
        println!("   Winning Trades:     {:>12} ({:.1}%)", winning_trades.len(), win_rate);
        println!("   Losing Trades:      {:>12} ({:.1}%)", losing_trades.len(), 100.0 - win_rate);
        println!();
        println!("   Gross Profit:       ${:>12.2}", gross_profit);
        println!("   Gross Loss:         ${:>12.2}", gross_loss);
        println!("   Profit Factor:      {:>12.2}", profit_factor);
        println!();
        println!("   Avg Win:            ${:>12.2}", avg_win);
        println!("   Avg Loss:           ${:>12.2}", avg_loss);
        println!();

        println!("🎯 EXIT ANALYSIS");
        println!("   ─────────────────────────────────────");
        println!("   Stop Loss Exits:    {:>12} ({:.1}%)", sl_exits, sl_exits as f32 / total_trades.max(1) as f32 * 100.0);
        println!("   Take Profit Exits:  {:>12} ({:.1}%)", tp_exits, tp_exits as f32 / total_trades.max(1) as f32 * 100.0);
        println!();

        // Calculate max drawdown
        let mut peak = self.initial_balance;
        let mut max_dd = 0.0f32;
        let mut running_balance = self.initial_balance;
        for trade in &self.trades {
            running_balance += trade.pnl.unwrap_or(0.0);
            peak = peak.max(running_balance);
            let dd = (peak - running_balance) / peak;
            max_dd = max_dd.max(dd);
        }

        println!("⚠️  RISK METRICS");
        println!("   ─────────────────────────────────────");
        println!("   Max Drawdown:       {:>12.2}%", max_dd * 100.0);

        // Sharpe ratio approximation (assuming daily returns)
        if total_trades > 1 {
            let returns: Vec<f32> = self.trades.iter().map(|t| t.pnl.unwrap_or(0.0) / self.initial_balance).collect();
            let avg_return = returns.iter().sum::<f32>() / returns.len() as f32;
            let variance: f32 = returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f32>() / returns.len() as f32;
            let std_dev = variance.sqrt();
            let sharpe = if std_dev > 0.0 { avg_return / std_dev * (252_f32).sqrt() } else { 0.0 };
            println!("   Sharpe Ratio (ann): {:>12.2}", sharpe);
        }

        println!();
        // Get leverage from first trade (all trades use same leverage)
        let leverage = self.trades.first().map(|t| t.leverage).unwrap_or(1.0);

        println!("⚙️  STRATEGY SETTINGS");
        println!("   ─────────────────────────────────────");
        println!("   Min Confidence:     {:>12.1}%", MIN_CONFIDENCE * 100.0);
        println!("   Risk per Trade:     {:>12.1}%", RISK_PER_TRADE * 100.0);
        println!("   Stop Loss:          {:>12.2}%", STOP_LOSS_PCT * 100.0);
        println!("   Take Profit:        {:>12.2}%", TAKE_PROFIT_PCT * 100.0);
        println!("   Trading Fee:        {:>12.3}%", FEE_PCT * 100.0);
        println!("   Leverage:           {:>12.1}x", leverage);
        println!();

        // Count liquidations
        let liquidations = self.trades.iter()
            .filter(|t| t.exit_reason.as_deref() == Some("liquidation"))
            .count();
        if liquidations > 0 {
            println!("💀 LIQUIDATIONS:       {:>12}", liquidations);
            println!();
        }
    }

    fn export_trades(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.trades)?;
        std::fs::write(path, json)?;
        println!("📁 Trades exported to: {}", path);
        Ok(())
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let start_time = Instant::now();

    // Parse arguments
    let args: Vec<String> = env::args().collect();
    let mut model_path = "data/charter_model.onnx".to_string();
    let mut data_path = "data/btc.csv".to_string();
    let mut initial_balance = 10000.0f32;
    let mut leverage = LEVERAGE;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                model_path = args[i + 1].clone();
                i += 2;
            }
            "--data" if i + 1 < args.len() => {
                data_path = args[i + 1].clone();
                i += 2;
            }
            "--initial-balance" if i + 1 < args.len() => {
                initial_balance = args[i + 1].parse().unwrap_or(10000.0);
                i += 2;
            }
            "--leverage" if i + 1 < args.len() => {
                leverage = args[i + 1].parse::<f32>().unwrap_or(1.0).clamp(1.0, 100.0);
                i += 2;
            }
            _ => i += 1,
        }
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║              Charter ML Backtest Engine                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Load model
    print!("📦 Loading ML model... ");
    let ml_inference = MlInferenceHandle::load(&model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
    println!("✓");

    // Load data
    print!("📂 Loading price data... ");
    let all_candles = load_candles_from_csv(&data_path)?;
    println!("✓ {} candles", all_candles.len());

    // Use last 1.5M candles for backtest (~1042 days = ~149 weeks)
    // This ensures we have enough 1w candles (>100 needed for TA)
    let max_candles = 1_500_000;
    let candles: Vec<Candle> = if all_candles.len() > max_candles {
        let skip = all_candles.len() - max_candles;
        all_candles[skip..].to_vec()
    } else {
        all_candles
    };

    // Aggregate to timeframes
    print!("📊 Aggregating timeframes... ");
    let timeframes = vec![
        Timeframe::Min1,
        Timeframe::Min3,
        Timeframe::Min5,
        Timeframe::Min30,
        Timeframe::Hour1,
        Timeframe::Hour3,
        Timeframe::Hour5,
        Timeframe::Hour10,
        Timeframe::Day1,
        Timeframe::Week1,
    ];
    let mut timeframe_candles: Vec<Vec<Candle>> = Vec::new();
    for tf in &timeframes {
        timeframe_candles.push(aggregate_candles(&candles, *tf));
    }
    println!("✓");

    // Run TA for each ML timeframe
    print!("📈 Computing TA for ML timeframes... ");
    let mut ta_data: Vec<(Vec<charter_ta::Level>, Vec<charter_ta::Trend>)> = vec![(vec![], vec![]); timeframes.len()];

    for &tf_idx in &ML_TIMEFRAME_INDICES {
        if tf_idx >= timeframe_candles.len() {
            eprintln!("   Skipping tf_idx {} - out of range", tf_idx);
            continue;
        }
        let tf_candles = &timeframe_candles[tf_idx];
        if tf_candles.len() < MIN_CANDLES {
            eprintln!("   Skipping tf_idx {} - only {} candles (need {})", tf_idx, tf_candles.len(), MIN_CANDLES);
            continue;
        }

        let config = AnalyzerConfig::default();
        let mut analyzer = Analyzer::with_config(config);
        for candle in tf_candles {
            analyzer.process_candle(*candle);
        }
        let levels = analyzer.all_levels().to_vec();
        let trends = analyzer.all_trends().to_vec();
        eprintln!("   tf_idx {}: {} candles -> {} levels, {} trends", tf_idx, tf_candles.len(), levels.len(), trends.len());
        ta_data[tf_idx] = (levels, trends);
    }
    println!("✓");

    // Initialize backtest engine
    let mut engine = BacktestEngine::new(initial_balance, ml_inference);

    // Use 5m as primary timeframe for trading
    let primary_tf_idx = 2; // 5m
    let primary_candles = &timeframe_candles[primary_tf_idx];

    println!();
    println!("🚀 Running backtest on {} candles (~{} days)...",
        primary_candles.len(),
        primary_candles.len() / 288
    );
    println!();

    // Progress tracking
    let total = primary_candles.len();
    let mut last_pct = 0;
    let mut features_extracted = 0;
    let mut features_failed = 0;

    // Skip warmup period and trade every N candles for speed
    let warmup = MIN_CANDLES * 2;
    let trade_interval = 5; // Only consider trades every 5 candles (25 min on 5m)

    for (idx, candle) in primary_candles.iter().enumerate() {
        // Skip warmup
        if idx < warmup {
            continue;
        }

        // Always check for exits on open trades
        if engine.current_trade.is_some() {
            engine.process_candle(candle, None, leverage)?;
        }

        // Only try to open new trades every N candles
        if idx % trade_interval == 0 && engine.current_trade.is_none() {
            let features = extract_features(
                candle,
                idx,
                &timeframe_candles,
                &ta_data,
                primary_candles,
            );

            if features.is_some() {
                features_extracted += 1;
                engine.process_candle(candle, features.as_ref(), leverage)?;
            } else {
                features_failed += 1;
            }
        }

        // Progress with bankroll
        let pct = idx * 100 / total;
        if pct > last_pct && pct % 5 == 0 {
            let trades = engine.trades.len();
            let open = if engine.current_trade.is_some() { " [OPEN]" } else { "" };
            print!("\r   Progress: {:>3}% | Balance: ${:>10.2} | Trades: {:>4}{}",
                   pct, engine.balance, trades, open);
            std::io::Write::flush(&mut std::io::stdout())?;
            last_pct = pct;
        }
    }

    // Close any open trade at end
    if let Some(last_candle) = primary_candles.last() {
        engine.close_open_trade(last_candle);
    }

    println!("\r   Progress: 100%");
    println!();
    println!("📊 Feature extraction: {} succeeded, {} failed", features_extracted, features_failed);
    println!("   Total predictions attempted: {}", engine.total_predictions);

    // Print results
    engine.print_results();

    // Export trades for visualization
    engine.export_trades("data/backtest_trades.json")?;

    let elapsed = start_time.elapsed();
    println!("⏱️  Backtest completed in {:.2}s", elapsed.as_secs_f32());
    println!();

    Ok(())
}

/// Calculate RSI (Relative Strength Index) for a given candle index.
/// Returns value normalized to 0-1 range (typical RSI is 0-100, so we divide by 100).
fn calculate_rsi(candles: &[Candle], current_idx: usize, period: usize) -> f32 {
    if current_idx < period + 1 || candles.is_empty() {
        return 0.5; // Neutral if not enough data
    }

    let start_idx = current_idx.saturating_sub(100.min(current_idx)); // Look back max 100 candles
    let lookback_candles = &candles[start_idx..=current_idx];

    if lookback_candles.len() < period + 1 {
        return 0.5;
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    // Calculate price changes
    for i in 1..lookback_candles.len() {
        let change = lookback_candles[i].close - lookback_candles[i - 1].close;
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }

    if gains.len() < period {
        return 0.5;
    }

    // Calculate average gain and loss using Wilder's smoothing method
    let mut avg_gain: f32 = gains.iter().take(period).sum::<f32>() / period as f32;
    let mut avg_loss: f32 = losses.iter().take(period).sum::<f32>() / period as f32;

    // Apply smoothing for remaining values
    for i in period..gains.len() {
        avg_gain = (avg_gain * (period - 1) as f32 + gains[i]) / period as f32;
        avg_loss = (avg_loss * (period - 1) as f32 + losses[i]) / period as f32;
    }

    // Calculate RSI
    if avg_loss == 0.0 {
        return 1.0; // Maximum RSI (100) normalized to 1.0
    }

    let rs = avg_gain / avg_loss;
    let rsi = 100.0 - (100.0 / (1.0 + rs));

    // Normalize to 0-1 range
    rsi / 100.0
}

/// Extract ML features for a candle.
fn extract_features(
    candle: &Candle,
    candle_idx: usize,
    timeframe_candles: &[Vec<Candle>],
    ta_data: &[(Vec<charter_ta::Level>, Vec<charter_ta::Trend>)],
    primary_candles: &[Candle],
) -> Option<MlFeatures> {
    let current_timestamp = candle.timestamp;
    let current_price = candle.close;

    let mut tf_features: Vec<TimeframeFeatures> = Vec::new();

    for (feature_idx, &tf_idx) in ML_TIMEFRAME_INDICES.iter().enumerate() {
        if tf_idx >= timeframe_candles.len() {
            continue;
        }

        let candles = &timeframe_candles[tf_idx];
        let (levels, trends) = &ta_data[tf_idx];

        if candles.len() < MIN_CANDLES || levels.is_empty() {
            continue;
        }

        // Find candle index for this timestamp
        let tf_candle_idx = find_candle_index(candles, current_timestamp)?;

        // Filter levels/trends created before this point
        let active_levels: Vec<_> = levels
            .iter()
            .filter(|l| l.created_at_index <= tf_candle_idx)
            .cloned()
            .collect();
        let active_trends: Vec<_> = trends
            .iter()
            .filter(|t| t.created_at_index <= tf_candle_idx)
            .cloned()
            .collect();

        let features = TimeframeFeatures::extract(
            feature_idx,
            &active_levels,
            &active_trends,
            current_price,
            tf_candle_idx,
        );
        tf_features.push(features);
    }

    // Need all 4 timeframes for the model (302 features = 4 × 74 + 6 with RSI)
    if tf_features.len() != 4 {
        return None;
    }

    // Calculate global features
    let prev_close = if candle_idx > 0 {
        primary_candles[candle_idx - 1].close
    } else {
        candle.open
    };
    let price_change = if prev_close > 0.0 {
        (candle.close - prev_close) / prev_close
    } else {
        0.0
    };

    let body = (candle.close - candle.open).abs();
    let range = candle.high - candle.low;
    let body_ratio = if range > 0.0 { body / range } else { 0.5 };

    let lookback = 100.min(candle_idx);
    let avg_volume: f32 = if lookback > 0 {
        primary_candles[candle_idx - lookback..candle_idx]
            .iter()
            .map(|c| c.volume)
            .sum::<f32>()
            / lookback as f32
    } else {
        candle.volume
    };
    let volume_normalized = if avg_volume > 0.0 {
        candle.volume / avg_volume
    } else {
        1.0
    };

    // Calculate RSI (14-period)
    let rsi_14 = calculate_rsi(primary_candles, candle_idx, 14);

    Some(MlFeatures {
        timeframes: tf_features,
        current_price,
        current_volume_normalized: volume_normalized,
        price_change_normalized: price_change,
        body_ratio,
        is_bullish: if candle.close > candle.open { 1.0 } else { 0.0 },
        rsi_14,
    })
}

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

    if lo > 0 && (candles[lo].timestamp - timestamp).abs() > (candles[lo - 1].timestamp - timestamp).abs() {
        Some(lo - 1)
    } else {
        Some(lo)
    }
}

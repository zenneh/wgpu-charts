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
use charter_core::Candle;
use charter_data::load_candles_from_csv;
use charter_ta::{
    AnalyzerConfig, MlFeatures, MlInferenceHandle, MultiTimeframeAnalyzer,
    ml_export_timeframes,
};

// ============================================================================
// Configuration - Scalping Strategy
// ============================================================================

/// Take profit as percentage of entry price.
const SCALP_TP_PCT: f32 = 0.008; // 0.8% take profit

/// Stop loss as percentage of entry price.
const SCALP_SL_PCT: f32 = 0.005; // 0.5% stop loss (1.6:1 R:R)

/// Proximity threshold to consider price "near" a level.
const LEVEL_PROXIMITY_PCT: f32 = 0.02; // 2% proximity to level

/// Maximum candles to hold a trade before timeout exit.
const MAX_HOLD_CANDLES: usize = 30; // 30 minutes to reach target

/// Profit threshold to move stop loss to breakeven.
const BREAKEVEN_TRIGGER_PCT: f32 = 0.004; // 0.4% (move to BE after 50% of TP target)

/// Minimum body ratio for breakout signals (body / range).
const MIN_BODY_RATIO: f32 = 0.6; // 60% body for breakouts

/// Minimum body ratio for bounce signals (more lenient).
const MIN_BOUNCE_BODY_RATIO: f32 = 0.4; // 40% body for bounces

/// Trading fee per trade (taker fee).
const FEE_PCT: f32 = 0.0001; // 0.01%

/// Leverage multiplier for scalping.
const LEVERAGE: f32 = 50.0; // Moderate leverage (reduced from 100x to survive volatility)

/// Risk per trade as fraction of bankroll.
const RISK_PER_TRADE: f32 = 0.01; // 1% risk per trade

/// ML threshold for taking long positions (hybrid approach).
const ML_LONG_THRESHOLD: f32 = 0.48; // Take longs if direction_up_prob > 0.48

/// ML threshold for taking short positions (hybrid approach).
const ML_SHORT_THRESHOLD: f32 = 0.52; // Take shorts if direction_up_prob < 0.52

/// Cooldown candles after a trade closes before entering a new one.
const TRADE_COOLDOWN_CANDLES: usize = 5;

/// Maximum drawdown before stopping (20% below initial balance).
const MAX_DRAWDOWN_PCT: f32 = 0.20; // Stop if balance drops 20% below initial

/// Minimum candles for TA warmup.
#[allow(dead_code)]
const MIN_CANDLES: usize = 100;

// ============================================================================
// Trading Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
enum Direction {
    Long,
    Short,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
enum ScalpSignal {
    SupportBounce,    // Long near support
    ResistanceBounce, // Short near resistance
    ResistanceBreak,  // Long breakout above resistance
    SupportBreak,     // Short breakdown below support
}

impl std::fmt::Display for ScalpSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalpSignal::SupportBounce => write!(f, "SupportBounce"),
            ScalpSignal::ResistanceBounce => write!(f, "ResistanceBounce"),
            ScalpSignal::ResistanceBreak => write!(f, "ResistanceBreak"),
            ScalpSignal::SupportBreak => write!(f, "SupportBreak"),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct Trade {
    direction: Direction,
    signal_type: ScalpSignal,
    entry_price: f32,
    entry_time: f64,
    entry_candle_index: usize,
    margin: f32, // Margin (collateral) in USD
    size: f32,   // Position size in USD (margin * leverage)
    leverage: f32,
    stop_loss: f32,
    take_profit: f32,
    liquidation_price: f32,
    is_breakeven: bool,
    exit_price: Option<f32>,
    exit_time: Option<f64>,
    pnl: Option<f32>,
    exit_reason: Option<String>,
}

impl Trade {
    fn new(
        direction: Direction,
        signal_type: ScalpSignal,
        entry_price: f32,
        entry_time: f64,
        entry_candle_index: usize,
        margin: f32,
        leverage: f32,
    ) -> Self {
        let size = margin * leverage;

        // Scalping uses tight stops
        let sl_pct = SCALP_SL_PCT;
        let tp_pct = SCALP_TP_PCT;

        let (stop_loss, take_profit, liquidation_price) = match direction {
            Direction::Long => (
                entry_price * (1.0 - sl_pct),
                entry_price * (1.0 + tp_pct),
                // Liquidation when loss >= margin (e.g., at 100x, 1% drop = liquidation)
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
            signal_type,
            entry_price,
            entry_time,
            entry_candle_index,
            margin,
            size,
            leverage,
            stop_loss,
            take_profit,
            liquidation_price,
            is_breakeven: false,
            exit_price: None,
            exit_time: None,
            pnl: None,
            exit_reason: None,
        }
    }

    fn check_exit(&mut self, candle: &Candle, current_index: usize) -> bool {
        // First update breakeven stop if applicable
        self.update_breakeven(candle.close);

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
            let reason = if self.is_breakeven { "breakeven" } else { "stop_loss" };
            self.close(self.stop_loss, candle.timestamp, reason);
            return true;
        }

        if hit_tp {
            self.close(self.take_profit, candle.timestamp, "take_profit");
            return true;
        }

        // Time-based exit: close trade if held too long
        if current_index - self.entry_candle_index >= MAX_HOLD_CANDLES {
            self.close(candle.close, candle.timestamp, "timeout");
            return true;
        }

        false
    }

    fn update_breakeven(&mut self, current_price: f32) {
        if self.is_breakeven {
            return;
        }

        let profit_pct = match self.direction {
            Direction::Long => (current_price - self.entry_price) / self.entry_price,
            Direction::Short => (self.entry_price - current_price) / self.entry_price,
        };

        if profit_pct >= BREAKEVEN_TRIGGER_PCT {
            self.stop_loss = self.entry_price;
            self.is_breakeven = true;
        }
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
    #[allow(dead_code)]
    correct_predictions: usize,
    debug_trade_count: usize,

    // Cooldown tracking
    last_trade_exit_index: Option<usize>,
}

const DEBUG_TRADES_TO_PRINT: usize = 10;

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
            debug_trade_count: 0,
            last_trade_exit_index: None,
        }
    }

    fn process_candle(
        &mut self,
        candle: &Candle,
        features: Option<&MlFeatures>,
        leverage: f32,
        current_index: usize,
    ) -> Result<()> {
        // Check if current trade should be closed
        if let Some(ref mut trade) = self.current_trade {
            if trade.check_exit(candle, current_index) {
                let pnl = trade.pnl.unwrap_or(0.0);
                self.balance += pnl;
                self.trades.push(trade.clone());
                self.current_trade = None;
                self.last_trade_exit_index = Some(current_index);
            }
        }

        // Check cooldown period
        let in_cooldown = self.last_trade_exit_index
            .map(|exit_idx| current_index < exit_idx + TRADE_COOLDOWN_CANDLES)
            .unwrap_or(false);

        // Try to open new scalp trade if no position and not in cooldown
        if self.current_trade.is_none() && !in_cooldown {
            if let Some(features) = features {
                self.try_scalp_entry(candle, features, leverage, current_index)?;
            }
        }

        Ok(())
    }

    fn try_scalp_entry(
        &mut self,
        candle: &Candle,
        features: &MlFeatures,
        leverage: f32,
        current_index: usize,
    ) -> Result<()> {
        // Get ML prediction for direction bias
        let prediction = match self.ml_inference.predict(features) {
            Ok(p) => p,
            Err(_) => return Ok(()),
        };

        self.total_predictions += 1;
        let ml_direction_up_prob = prediction.direction_up_prob;

        // Try to detect scalp entry signal using level proximity + ML direction
        let signal = match Self::detect_scalp_signal(candle, features, ml_direction_up_prob) {
            Some(s) => s,
            None => return Ok(()),
        };

        let direction = match signal {
            ScalpSignal::SupportBounce | ScalpSignal::ResistanceBreak => Direction::Long,
            ScalpSignal::ResistanceBounce | ScalpSignal::SupportBreak => Direction::Short,
        };

        // Calculate margin (collateral) based on risk
        // Margin = (Balance * Risk%) / (StopLoss% * Leverage)
        let margin = (self.balance * RISK_PER_TRADE) / (SCALP_SL_PCT * leverage);
        let margin = margin.min(self.balance * 0.5); // Max 50% of balance as margin

        // Minimum margin: 0.1% of balance (allows small starting balances)
        let min_margin = self.balance * 0.001;
        if margin < min_margin.max(0.01) {
            return Ok(()); // Skip if margin is too small
        }

        // Debug output for first few trades
        if self.debug_trade_count < DEBUG_TRADES_TO_PRINT {
            // Calculate actual support/resistance distances (corrected interpretation)
            let actual_support_dist = features.closest_resistance_distance
                .filter(|&d| d < 0.0).map(|d| d.abs())
                .or_else(|| features.closest_support_distance.filter(|&d| d > 0.0));
            let actual_resistance_dist = features.closest_support_distance
                .filter(|&d| d < 0.0).map(|d| d.abs())
                .or_else(|| features.closest_resistance_distance.filter(|&d| d > 0.0));

            println!("\n📈 Trade #{} Entry Debug:", self.debug_trade_count + 1);
            println!("   Signal: {:?} -> {:?}", signal, direction);
            println!("   Price: ${:.2}", candle.close);
            println!("   ML up_prob: {:.3}", ml_direction_up_prob);
            println!("   Support (below): {:?}", actual_support_dist.map(|d| format!("{:.2}%", d * 100.0)));
            println!("   Resistance (above): {:?}", actual_resistance_dist.map(|d| format!("{:.2}%", d * 100.0)));
            println!("   Body ratio: {:.2}, Bullish: {}",
                (candle.close - candle.open).abs() / (candle.high - candle.low),
                candle.close > candle.open);
            self.debug_trade_count += 1;
        }

        // Open scalp trade
        let trade = Trade::new(
            direction,
            signal,
            candle.close,
            candle.timestamp,
            current_index,
            margin,
            leverage,
        );
        self.current_trade = Some(trade);

        Ok(())
    }

    fn detect_scalp_signal(
        candle: &Candle,
        features: &MlFeatures,
        ml_direction_up_prob: f32,
    ) -> Option<ScalpSignal> {
        let range = candle.high - candle.low;
        if range < f32::EPSILON {
            return None;
        }

        let body_ratio = (candle.close - candle.open).abs() / range;
        let is_bullish = candle.close > candle.open;

        // IMPORTANT: The level distances from charter-ta are INVERTED!
        // - closest_resistance_distance < 0 means the level is BELOW price (acts as SUPPORT)
        // - closest_support_distance < 0 means the level is ABOVE price (acts as RESISTANCE)
        //
        // Correct interpretation:
        // - Level BELOW price (dist < 0 for resistance, dist > 0 for support) = Support zone
        // - Level ABOVE price (dist > 0 for resistance, dist < 0 for support) = Resistance zone

        // Find actual support (level below current price)
        let support_level_dist = features.closest_resistance_distance
            .filter(|&d| d < 0.0)  // resistance_dist < 0 means level is below
            .map(|d| d.abs())
            .or_else(|| features.closest_support_distance
                .filter(|&d| d > 0.0)  // support_dist > 0 means level is below
            );

        // Find actual resistance (level above current price)
        let resistance_level_dist = features.closest_support_distance
            .filter(|&d| d < 0.0)  // support_dist < 0 means level is above
            .map(|d| d.abs())
            .or_else(|| features.closest_resistance_distance
                .filter(|&d| d > 0.0)  // resistance_dist > 0 means level is above
            );

        // BOUNCE SIGNALS: Enter near level with ML confirmation

        // Support bounce (long) - price near support (below) + ML bullish + bullish candle
        if let Some(dist) = support_level_dist {
            if dist < LEVEL_PROXIMITY_PCT
                && ml_direction_up_prob > ML_LONG_THRESHOLD
                && is_bullish
                && body_ratio > MIN_BOUNCE_BODY_RATIO
            {
                return Some(ScalpSignal::SupportBounce);
            }
        }

        // Resistance bounce (short) - price near resistance (above) + ML bearish + bearish candle
        if let Some(dist) = resistance_level_dist {
            if dist < LEVEL_PROXIMITY_PCT
                && ml_direction_up_prob < ML_SHORT_THRESHOLD
                && !is_bullish
                && body_ratio > MIN_BOUNCE_BODY_RATIO
            {
                return Some(ScalpSignal::ResistanceBounce);
            }
        }

        // BREAK SIGNALS: Strong momentum through level with ML confirmation

        // Resistance break (long) - strong bullish candle breaking through resistance above
        if let Some(dist) = resistance_level_dist {
            if dist < LEVEL_PROXIMITY_PCT * 2.0
                && is_bullish
                && body_ratio > MIN_BODY_RATIO
                && ml_direction_up_prob > 0.5
            {
                return Some(ScalpSignal::ResistanceBreak);
            }
        }

        // Support break (short) - strong bearish candle breaking through support below
        if let Some(dist) = support_level_dist {
            if dist < LEVEL_PROXIMITY_PCT * 2.0
                && !is_bullish
                && body_ratio > MIN_BODY_RATIO
                && ml_direction_up_prob < 0.5
            {
                return Some(ScalpSignal::SupportBreak);
            }
        }

        None
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
        let winning_trades: Vec<_> = self
            .trades
            .iter()
            .filter(|t| t.pnl.unwrap_or(0.0) > 0.0)
            .collect();
        let losing_trades: Vec<_> = self
            .trades
            .iter()
            .filter(|t| t.pnl.unwrap_or(0.0) <= 0.0)
            .collect();

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
        let sl_exits = self
            .trades
            .iter()
            .filter(|t| t.exit_reason.as_deref() == Some("stop_loss"))
            .count();
        let tp_exits = self
            .trades
            .iter()
            .filter(|t| t.exit_reason.as_deref() == Some("take_profit"))
            .count();

        println!("📊 PERFORMANCE SUMMARY");
        println!("   ─────────────────────────────────────");
        println!("   Initial Balance:    ${:>12.2}", self.initial_balance);
        println!("   Final Balance:      ${:>12.2}", self.balance);
        println!(
            "   Total P&L:          ${:>12.2} ({:+.2}%)",
            total_pnl, return_pct
        );
        println!();

        println!("📈 TRADE STATISTICS");
        println!("   ─────────────────────────────────────");
        println!("   Total Trades:       {:>12}", total_trades);
        println!(
            "   Winning Trades:     {:>12} ({:.1}%)",
            winning_trades.len(),
            win_rate
        );
        println!(
            "   Losing Trades:      {:>12} ({:.1}%)",
            losing_trades.len(),
            100.0 - win_rate
        );
        println!();
        println!("   Gross Profit:       ${:>12.2}", gross_profit);
        println!("   Gross Loss:         ${:>12.2}", gross_loss);
        println!("   Profit Factor:      {:>12.2}", profit_factor);
        println!();
        println!("   Avg Win:            ${:>12.2}", avg_win);
        println!("   Avg Loss:           ${:>12.2}", avg_loss);
        println!();

        // Additional scalping exit reasons
        let timeout_exits = self
            .trades
            .iter()
            .filter(|t| t.exit_reason.as_deref() == Some("timeout"))
            .count();
        let breakeven_exits = self
            .trades
            .iter()
            .filter(|t| t.exit_reason.as_deref() == Some("breakeven"))
            .count();

        println!("🎯 EXIT ANALYSIS");
        println!("   ─────────────────────────────────────");
        println!(
            "   Stop Loss Exits:    {:>12} ({:.1}%)",
            sl_exits,
            sl_exits as f32 / total_trades.max(1) as f32 * 100.0
        );
        println!(
            "   Take Profit Exits:  {:>12} ({:.1}%)",
            tp_exits,
            tp_exits as f32 / total_trades.max(1) as f32 * 100.0
        );
        println!(
            "   Timeout Exits:      {:>12} ({:.1}%)",
            timeout_exits,
            timeout_exits as f32 / total_trades.max(1) as f32 * 100.0
        );
        println!(
            "   Breakeven Exits:    {:>12} ({:.1}%)",
            breakeven_exits,
            breakeven_exits as f32 / total_trades.max(1) as f32 * 100.0
        );
        let max_dd_exits = self
            .trades
            .iter()
            .filter(|t| t.exit_reason.as_deref() == Some("max_drawdown"))
            .count();
        if max_dd_exits > 0 {
            println!(
                "   Max Drawdown Exit:  {:>12} ({:.1}%)",
                max_dd_exits,
                max_dd_exits as f32 / total_trades.max(1) as f32 * 100.0
            );
        }
        println!();

        // Signal type breakdown
        let support_bounce = self
            .trades
            .iter()
            .filter(|t| t.signal_type == ScalpSignal::SupportBounce)
            .count();
        let resistance_bounce = self
            .trades
            .iter()
            .filter(|t| t.signal_type == ScalpSignal::ResistanceBounce)
            .count();
        let resistance_break = self
            .trades
            .iter()
            .filter(|t| t.signal_type == ScalpSignal::ResistanceBreak)
            .count();
        let support_break = self
            .trades
            .iter()
            .filter(|t| t.signal_type == ScalpSignal::SupportBreak)
            .count();

        println!("📍 SIGNAL BREAKDOWN");
        println!("   ─────────────────────────────────────");
        println!(
            "   Support Bounce:     {:>12} ({:.1}%)",
            support_bounce,
            support_bounce as f32 / total_trades.max(1) as f32 * 100.0
        );
        println!(
            "   Resistance Bounce:  {:>12} ({:.1}%)",
            resistance_bounce,
            resistance_bounce as f32 / total_trades.max(1) as f32 * 100.0
        );
        println!(
            "   Resistance Break:   {:>12} ({:.1}%)",
            resistance_break,
            resistance_break as f32 / total_trades.max(1) as f32 * 100.0
        );
        println!(
            "   Support Break:      {:>12} ({:.1}%)",
            support_break,
            support_break as f32 / total_trades.max(1) as f32 * 100.0
        );
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
            let returns: Vec<f32> = self
                .trades
                .iter()
                .map(|t| t.pnl.unwrap_or(0.0) / self.initial_balance)
                .collect();
            let avg_return = returns.iter().sum::<f32>() / returns.len() as f32;
            let variance: f32 = returns
                .iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f32>()
                / returns.len() as f32;
            let std_dev = variance.sqrt();
            let sharpe = if std_dev > 0.0 {
                avg_return / std_dev * (252_f32).sqrt()
            } else {
                0.0
            };
            println!("   Sharpe Ratio (ann): {:>12.2}", sharpe);
        }

        println!();
        // Get leverage from first trade (all trades use same leverage)
        let leverage = self.trades.first().map(|t| t.leverage).unwrap_or(1.0);

        println!("⚙️  SCALPING SETTINGS");
        println!("   ─────────────────────────────────────");
        println!("   Risk per Trade:     {:>12.1}%", RISK_PER_TRADE * 100.0);
        println!("   Stop Loss:          {:>12.3}%", SCALP_SL_PCT * 100.0);
        println!("   Take Profit:        {:>12.3}%", SCALP_TP_PCT * 100.0);
        println!("   Level Proximity:    {:>12.3}%", LEVEL_PROXIMITY_PCT * 100.0);
        println!("   Breakeven Trigger:  {:>12.3}%", BREAKEVEN_TRIGGER_PCT * 100.0);
        println!("   Max Hold Candles:   {:>12}", MAX_HOLD_CANDLES);
        println!("   Trading Fee:        {:>12.4}%", FEE_PCT * 100.0);
        println!("   Leverage:           {:>12.1}x", leverage);
        println!("   Max Drawdown Stop:  {:>12.1}%", MAX_DRAWDOWN_PCT * 100.0);
        println!("   ML Long Threshold:  {:>12.2}", ML_LONG_THRESHOLD);
        println!("   ML Short Threshold: {:>12.2}", ML_SHORT_THRESHOLD);
        println!();

        // Count liquidations
        let liquidations = self
            .trades
            .iter()
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
    let mut initial_balance = 100.0f32;
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
                initial_balance = args[i + 1].parse().unwrap_or(10.0);
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

    // Use last N candles for backtest
    let max_candles = 500_000; // ~347 days for comprehensive test
    let candles_1m: Vec<Candle> = if all_candles.len() > max_candles {
        let skip = all_candles.len() - max_candles;
        all_candles[skip..].to_vec()
    } else {
        all_candles
    };

    // Create incremental multi-timeframe analyzer (FAST - no upfront computation)
    print!("📊 Setting up incremental TA... ");
    let timeframes = ml_export_timeframes(); // 1m, 5m, 30m, 1h, 1d
    let config = AnalyzerConfig::default();
    let mut mta = MultiTimeframeAnalyzer::with_timeframes(timeframes.clone(), config);
    println!(
        "✓ ({} timeframes: {:?})",
        timeframes.len(),
        timeframes.iter().map(|t| t.label()).collect::<Vec<_>>()
    );

    // Initialize backtest engine
    let mut engine = BacktestEngine::new(initial_balance, ml_inference);

    // Debug: check first feature extraction
    let mut first_feature_debug = true;

    // Scalping: trade on every candle (no interval)
    let warmup_1m = 2000; // Warmup in 1m candles

    println!();
    println!(
        "🚀 Running scalping backtest on {} 1m candles (~{} days)...",
        candles_1m.len(),
        candles_1m.len() / 1440
    );
    println!("   Strategy: Level proximity + ML direction hybrid");
    println!("   TP: {:.2}% | SL: {:.2}% | Max hold: {} candles",
        SCALP_TP_PCT * 100.0, SCALP_SL_PCT * 100.0, MAX_HOLD_CANDLES);
    println!();

    // Progress tracking
    let total = candles_1m.len();
    let mut last_pct = 0;
    let mut features_extracted = 0;
    let mut features_failed = 0;

    // Process 1m candles incrementally
    for (idx, candle) in candles_1m.iter().enumerate() {
        // Update TA incrementally (FAST)
        mta.process_1m_candle(candle);

        // Skip warmup
        if idx < warmup_1m {
            continue;
        }

        // Check for max drawdown - stop trading if balance drops 20% below initial
        let drawdown = (initial_balance - engine.balance) / initial_balance;
        if drawdown >= MAX_DRAWDOWN_PCT {
            println!("\n\n🛑 MAX DRAWDOWN REACHED: {:.1}% loss - stopping backtest", drawdown * 100.0);
            // Close any open trade at current price
            if let Some(ref mut trade) = engine.current_trade {
                trade.close(candle.close, candle.timestamp, "max_drawdown");
                let pnl = trade.pnl.unwrap_or(0.0);
                engine.balance += pnl;
                engine.trades.push(trade.clone());
                engine.current_trade = None;
            }
            break;
        }

        // Always check for exits on open trades
        if engine.current_trade.is_some() {
            engine.process_candle(candle, None, leverage, idx)?;
        }

        // Try to open new scalp trades on every candle (no interval for scalping)
        if engine.current_trade.is_none() {
            // Extract features from current MTA state
            let features = extract_features_incremental(candle, &mta);

            if let Some(ref f) = features {
                // Debug first extraction
                if first_feature_debug {
                    let feature_vec = f.to_vec();
                    println!("\n📊 Feature Debug:");
                    println!("   Timeframes: {}", f.timeframes.len());
                    println!("   Feature vector size: {}", feature_vec.len());
                    println!("   Total active levels: {}", f.total_active_levels);
                    println!("   Has resistance: {}, Has support: {}", f.has_resistance_above, f.has_support_below);
                    println!("   Closest support dist: {:?}", f.closest_support_distance);
                    println!("   Closest resistance dist: {:?}", f.closest_resistance_distance);

                    // Try prediction to see error
                    match engine.ml_inference.predict(f) {
                        Ok(pred) => println!("   Prediction: up_prob={:.3}, conf={:.3}", pred.direction_up_prob, pred.confidence),
                        Err(e) => println!("   Prediction error: {}", e),
                    }
                    println!();
                    first_feature_debug = false;
                }

                if f.timeframes.len() >= 5 {
                    features_extracted += 1;
                    engine.process_candle(candle, features.as_ref(), leverage, idx)?;
                } else {
                    features_failed += 1;
                }
            } else {
                features_failed += 1;
            }
        }

        // Progress with bankroll
        let pct = idx * 100 / total;
        if pct > last_pct && pct % 5 == 0 {
            let trades = engine.trades.len();
            let open = if engine.current_trade.is_some() {
                " [OPEN]"
            } else {
                ""
            };
            print!(
                "\r   Progress: {:>3}% | Balance: ${:>10.2} | Trades: {:>4}{}",
                pct, engine.balance, trades, open
            );
            std::io::Write::flush(&mut std::io::stdout())?;
            last_pct = pct;
        }
    }

    // Close any open trade at end
    if let Some(last_candle) = candles_1m.last() {
        engine.close_open_trade(last_candle);
    }

    println!("\r   Progress: 100%");
    println!();
    println!(
        "📊 Feature extraction: {} succeeded, {} failed",
        features_extracted, features_failed
    );
    println!(
        "   Total predictions attempted: {}",
        engine.total_predictions
    );

    // Print TA stats
    println!("📈 TA Statistics:");
    for (tf, levels, ranges) in mta.ta_counts() {
        println!("   • {}: {} levels, {} ranges", tf.label(), levels, ranges);
    }

    // Print results
    engine.print_results();

    // Export trades for visualization
    engine.export_trades("data/backtest_trades.json")?;

    let elapsed = start_time.elapsed();
    println!("⏱️  Backtest completed in {:.2}s", elapsed.as_secs_f32());
    println!();

    Ok(())
}

// ============================================================================
// Incremental Indicator Calculators
// ============================================================================

/// Incremental RSI calculator using Wilder's smoothing.
#[allow(dead_code)]
struct RsiState {
    period: usize,
    avg_gain: f32,
    avg_loss: f32,
    prev_close: f32,
    count: usize,
    gains: Vec<f32>,
    losses: Vec<f32>,
}

#[allow(dead_code)]
impl RsiState {
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

    fn update(&mut self, close: f32) {
        if self.count == 0 {
            self.prev_close = close;
            self.count = 1;
            return;
        }

        let change = close - self.prev_close;
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };
        self.prev_close = close;

        if self.count <= self.period {
            self.gains.push(gain);
            self.losses.push(loss);
            self.count += 1;

            if self.count == self.period + 1 {
                self.avg_gain = self.gains.iter().sum::<f32>() / self.period as f32;
                self.avg_loss = self.losses.iter().sum::<f32>() / self.period as f32;
                self.gains.clear();
                self.losses.clear();
            }
        } else {
            self.avg_gain = (self.avg_gain * (self.period - 1) as f32 + gain) / self.period as f32;
            self.avg_loss = (self.avg_loss * (self.period - 1) as f32 + loss) / self.period as f32;
            self.count += 1;
        }
    }

    fn value(&self) -> f32 {
        if self.count <= self.period {
            return 0.5;
        }
        if self.avg_loss == 0.0 {
            return 1.0;
        }
        let rs = self.avg_gain / self.avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        rsi / 100.0
    }
}

/// Rolling volume average calculator.
#[allow(dead_code)]
struct VolumeAverage {
    window: Vec<f32>,
    window_size: usize,
    sum: f32,
    idx: usize,
    filled: bool,
}

#[allow(dead_code)]
impl VolumeAverage {
    fn new(window_size: usize) -> Self {
        Self {
            window: vec![0.0; window_size],
            window_size,
            sum: 0.0,
            idx: 0,
            filled: false,
        }
    }

    fn update(&mut self, volume: f32) {
        self.sum -= self.window[self.idx];
        self.window[self.idx] = volume;
        self.sum += volume;
        self.idx = (self.idx + 1) % self.window_size;
        if self.idx == 0 {
            self.filled = true;
        }
    }

    fn normalized(&self, current_volume: f32) -> f32 {
        let count = if self.filled {
            self.window_size
        } else {
            self.idx.max(1)
        };
        let avg = self.sum / count as f32;
        if avg > 0.0 { current_volume / avg } else { 1.0 }
    }
}

/// Extract ML features from incremental multi-timeframe analyzer.
fn extract_features_incremental(
    candle: &Candle,
    mta: &MultiTimeframeAnalyzer,
) -> Option<MlFeatures> {
    let current_price = candle.close;

    // Extract features from each timeframe
    let tf_features = mta.extract_all_features(current_price);

    // Need at least some timeframes
    if tf_features.is_empty() {
        return None;
    }

    // Get state for global features
    let state = mta.state();

    // Find closest resistance and support distances
    // Use consistent formula: (level_price - current_price) / current_price
    // Support below = negative distance, Resistance above = positive distance
    let closest_resistance_distance = state.closest_unbroken_resistance.map(|(price, _)| {
        (price - current_price) / current_price
    });
    let closest_support_distance = state.closest_unbroken_support.map(|(price, _)| {
        (price - current_price) / current_price // Fixed: was (current_price - price)
    });

    // Compute price action features from current candle
    let range = candle.high - candle.low;
    let (body_ratio, upper_wick_ratio, lower_wick_ratio) = if range > f32::EPSILON {
        let body = (candle.close - candle.open).abs();
        let upper_wick = candle.high - candle.open.max(candle.close);
        let lower_wick = candle.open.min(candle.close) - candle.low;
        (body / range, upper_wick / range, lower_wick / range)
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
        // Price action features
        body_ratio,
        upper_wick_ratio,
        lower_wick_ratio,
        is_bullish,
        // Momentum features (not available in incremental mode without candle history)
        // These default to 0.0 - the model should handle this gracefully
        price_change_1: 0.0,
        price_change_3: 0.0,
        price_change_5: 0.0,
    })
}

// ============================================================================
// Legacy RSI function (kept for reference)
// ============================================================================

/// Calculate RSI (Relative Strength Index) for a given candle index.
/// Returns value normalized to 0-1 range (typical RSI is 0-100, so we divide by 100).
#[allow(dead_code)]
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


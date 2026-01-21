#!/usr/bin/env python3
"""
Backtest the level hold/break model using the actual trained XGBoost model.

This script validates the model's predictions match actual trading outcomes.

Usage: python backtest_level_model.py <events_csv> <model_json> <scaler_json>
"""

import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Configuration
HOLD_THRESHOLD = 0.65   # Confidence to take bounce trade
BREAK_THRESHOLD = 0.35  # Confidence to take breakout trade
BOUNCE_TP = 0.002       # 0.2% take profit for bounce
BOUNCE_SL = 0.0015      # 0.15% stop loss for bounce
BREAKOUT_TP = 0.003     # 0.3% take profit for breakout
BREAKOUT_SL = 0.002     # 0.2% stop loss for breakout
FEE = 0.0004            # 0.04% fee per trade


def load_model_and_scaler(model_path: str, scaler_path: str):
    """Load the XGBoost model and scaler."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(scaler_path, 'r') as f:
        scaler_data = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_data['mean'])
    scaler.scale_ = np.array(scaler_data['scale'])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)

    return model, scaler


def simulate_trade(is_bounce: bool, actual_held: bool) -> tuple[float, str]:
    """
    Simulate a trade outcome.

    Returns (pnl_percent, reason)
    """
    if is_bounce:
        tp, sl = BOUNCE_TP, BOUNCE_SL
        # Bounce trade: we bet the level holds
        # If it held, we win (price bounced in our favor)
        if actual_held:
            return tp - FEE, "TP"
        else:
            return -sl - FEE, "SL"
    else:
        tp, sl = BREAKOUT_TP, BREAKOUT_SL
        # Breakout trade: we bet the level breaks
        # If it broke, we win (price continued through)
        if not actual_held:
            return tp - FEE, "TP"
        else:
            return -sl - FEE, "SL"


def main():
    if len(sys.argv) < 4:
        print("Usage: python backtest_level_model.py <events_csv> <model_json> <scaler_json>")
        sys.exit(1)

    events_path = sys.argv[1]
    model_path = sys.argv[2]
    scaler_path = sys.argv[3]

    print("=" * 60)
    print("     Level Hold/Break Model Backtest")
    print("=" * 60)
    print()

    # Load data
    print(f"Loading events from {events_path}...")
    df = pd.read_csv(events_path)
    print(f"  Loaded {len(df)} events")

    # Load model and scaler
    print(f"Loading model from {model_path}...")
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    print("  Model loaded")

    # Split data chronologically (use last 20% for testing)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    print(f"  Testing on {len(test_df)} events (last 20%)")

    # Prepare features
    X_test = test_df.drop('held', axis=1).values.astype(np.float32)
    y_test = test_df['held'].values.astype(np.int32)
    X_test_scaled = scaler.transform(X_test)

    # Get predictions
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Simulate trading
    print("\nSimulating trades...")
    print()

    balance = 1000.0
    initial_balance = balance
    trades = []

    bounce_wins = 0
    bounce_losses = 0
    breakout_wins = 0
    breakout_losses = 0

    for i, (prob, actual) in enumerate(zip(y_prob, y_test)):
        trade_type = None
        is_bounce = False

        if prob >= HOLD_THRESHOLD:
            # High confidence level will hold -> bounce trade
            trade_type = "bounce"
            is_bounce = True
        elif prob <= BREAK_THRESHOLD:
            # High confidence level will break -> breakout trade
            trade_type = "breakout"
            is_bounce = False

        if trade_type is None:
            continue

        pnl_pct, reason = simulate_trade(is_bounce, actual == 1)
        pnl = balance * pnl_pct
        balance += pnl

        trades.append({
            'type': trade_type,
            'prob': prob,
            'actual': 'held' if actual == 1 else 'broke',
            'pnl': pnl,
            'reason': reason,
        })

        if is_bounce:
            if pnl > 0:
                bounce_wins += 1
            else:
                bounce_losses += 1
        else:
            if pnl > 0:
                breakout_wins += 1
            else:
                breakout_losses += 1

    # Results
    print("=" * 60)
    print("                    RESULTS")
    print("=" * 60)
    print()

    total_trades = len(trades)
    total_wins = bounce_wins + breakout_wins
    total_losses = bounce_losses + breakout_losses
    win_rate = 100 * total_wins / total_trades if total_trades > 0 else 0
    return_pct = 100 * (balance - initial_balance) / initial_balance

    print(f"Initial Balance: ${initial_balance:.2f}")
    print(f"Final Balance:   ${balance:.2f}")
    print(f"Total P&L:       ${balance - initial_balance:.2f} ({return_pct:+.2f}%)")
    print()

    print(f"Total Trades:    {total_trades}")
    print(f"Wins:            {total_wins} ({win_rate:.1f}%)")
    print(f"Losses:          {total_losses} ({100 - win_rate:.1f}%)")
    print()

    # Breakdown by trade type
    bounce_total = bounce_wins + bounce_losses
    breakout_total = breakout_wins + breakout_losses

    if bounce_total > 0:
        bounce_wr = 100 * bounce_wins / bounce_total
        print(f"Bounce Trades:   {bounce_total} ({bounce_wr:.1f}% win rate)")
    if breakout_total > 0:
        breakout_wr = 100 * breakout_wins / breakout_total
        print(f"Breakout Trades: {breakout_total} ({breakout_wr:.1f}% win rate)")
    print()

    # Profit factor
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = sum(abs(t['pnl']) for t in trades if t['pnl'] < 0)
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print(f"Gross Profit:    ${gross_profit:.2f}")
    print(f"Gross Loss:      ${gross_loss:.2f}")
    print(f"Profit Factor:   {pf:.2f}")

    print()
    print("=" * 60)
    print(f"Hold >= {HOLD_THRESHOLD:.0%} | Break <= {BREAK_THRESHOLD:.0%}")
    print(f"Bounce: TP {BOUNCE_TP:.2%} SL {BOUNCE_SL:.2%} | Breakout: TP {BREAKOUT_TP:.2%} SL {BREAKOUT_SL:.2%}")
    print("=" * 60)

    # Show prediction distribution for context
    print("\nPrediction distribution on test set:")
    print(f"  Min: {y_prob.min():.3f}")
    print(f"  Max: {y_prob.max():.3f}")
    print(f"  Mean: {y_prob.mean():.3f}")
    print(f"  >= {HOLD_THRESHOLD}: {(y_prob >= HOLD_THRESHOLD).sum()} events")
    print(f"  <= {BREAK_THRESHOLD}: {(y_prob <= BREAK_THRESHOLD).sum()} events")


if __name__ == '__main__':
    main()

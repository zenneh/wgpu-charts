#!/usr/bin/env python3
"""
Visualize backtest trades on an interactive price chart.

Usage:
    python scripts/plot_trades.py [--trades FILE] [--data FILE] [--range START-END]

Examples:
    python scripts/plot_trades.py
    python scripts/plot_trades.py --range 0-100
    python scripts/plot_trades.py --range 50-150 --data data/eth.csv
"""

import json
import argparse
import pandas as pd
import numpy as np

# Use plotly for interactive charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime


def load_trades(path: str) -> list:
    """Load trades from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def load_price_data(path: str) -> pd.DataFrame:
    """Load price data from CSV."""
    df = pd.read_csv(path)
    # Handle various column name formats (case-insensitive)
    df.columns = df.columns.str.lower()

    print(f"CSV columns: {list(df.columns)}")

    # Find timestamp column
    ts_col = None
    for col in df.columns:
        if 'timestamp' in col or col == 'time':
            ts_col = col
            break

    if ts_col:
        ts = df[ts_col].copy()
        # Handle mixed formats: convert ms to seconds where needed
        # Timestamps > 1e12 are in milliseconds
        mask = ts > 1e12
        ts.loc[mask] = ts.loc[mask] / 1000
        df['datetime'] = pd.to_datetime(ts, unit='s')
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])
    else:
        raise ValueError(f"Could not find timestamp column in: {list(df.columns)}")

    # Sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    print(f"Price data: {df['datetime'].min()} to {df['datetime'].max()} ({len(df)} rows)")

    return df


def plot_trades_interactive(trades: list, prices: pd.DataFrame, trade_range: tuple = None):
    """Create interactive visualization of trades on price chart."""

    if not trades:
        print("No trades to plot")
        return

    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    trades_df['entry_datetime'] = pd.to_datetime(trades_df['entry_time'], unit='s')
    trades_df['exit_datetime'] = pd.to_datetime(trades_df['exit_time'], unit='s')

    # Sort trades by entry time
    trades_df = trades_df.sort_values('entry_datetime').reset_index(drop=True)

    # Apply trade range filter
    if trade_range:
        start_idx, end_idx = trade_range
        end_idx = min(end_idx, len(trades_df))
        trades_df = trades_df.iloc[start_idx:end_idx].copy()
        print(f"Showing trades {start_idx} to {end_idx} ({len(trades_df)} trades)")

    if trades_df.empty:
        print("No trades in specified range")
        return

    # Filter price data to trade period with some padding
    min_time = trades_df['entry_datetime'].min() - pd.Timedelta(hours=6)
    max_time = trades_df['exit_datetime'].max() + pd.Timedelta(hours=6)

    print(f"Trade time range: {min_time} to {max_time}")
    print(f"Price data range: {prices['datetime'].min()} to {prices['datetime'].max()}")

    price_mask = (prices['datetime'] >= min_time) & (prices['datetime'] <= max_time)
    prices_filtered = prices[price_mask].copy()

    print(f"Filtered price points: {len(prices_filtered)}")

    if prices_filtered.empty:
        print("WARNING: No price data found for the trade period!")
        print("Creating price line from trade entry/exit prices instead...")
        # Fallback: create price line from trade data
        price_points = []
        for _, trade in trades_df.iterrows():
            price_points.append({'datetime': trade['entry_datetime'], 'open': trade['entry_price'],
                               'high': trade['entry_price'], 'low': trade['entry_price'], 'close': trade['entry_price']})
            price_points.append({'datetime': trade['exit_datetime'], 'open': trade['exit_price'],
                               'high': trade['exit_price'], 'low': trade['exit_price'], 'close': trade['exit_price']})
        ohlc_data = pd.DataFrame(price_points).sort_values('datetime').set_index('datetime')
    else:
        # Resample to OHLC for candlestick chart
        prices_filtered = prices_filtered.set_index('datetime').sort_index()
        date_range = (max_time - min_time).days

        if date_range > 365:
            resample_freq = '1D'
        elif date_range > 90:
            resample_freq = '4h'
        elif date_range > 30:
            resample_freq = '1h'
        elif date_range > 7:
            resample_freq = '30min'
        else:
            resample_freq = '15min'

        print(f"Resampling to {resample_freq} candlesticks (date range: {date_range} days)")

        # Resample OHLC data
        ohlc_data = prices_filtered.resample(resample_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        print(f"Candlestick count: {len(ohlc_data)}")

    # Calculate cumulative PnL
    trades_df_sorted = trades_df.sort_values('exit_datetime')
    cumulative_pnl = trades_df_sorted['pnl'].cumsum()

    # Calculate drawdown
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    drawdown_pct = (drawdown / running_max.clip(lower=1)) * 100

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=('Price with Trade Entries/Exits', 'Equity Curve', 'Drawdown'),
        row_heights=[0.5, 0.25, 0.25]
    )

    # Plot 1: Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=ohlc_data.index,
            open=ohlc_data['open'],
            high=ohlc_data['high'],
            low=ohlc_data['low'],
            close=ohlc_data['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350',
        ),
        row=1, col=1
    )

    print(f"Price range: ${ohlc_data['low'].min():.2f} - ${ohlc_data['high'].max():.2f}")

    # Separate winning and losing trades for coloring
    winners = trades_df[trades_df['pnl'] > 0]
    losers = trades_df[trades_df['pnl'] <= 0]

    # Long entries (green triangles up)
    long_trades = trades_df[trades_df['direction'] == 'Long']
    if not long_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=long_trades['entry_datetime'],
                y=long_trades['entry_price'],
                mode='markers',
                name='Long Entry',
                marker=dict(symbol='triangle-up', size=14, color='#00cc00', line=dict(width=2, color='darkgreen')),
                text=[f"LONG ENTRY #{i}<br>Price: ${p:.2f}<br>Margin: ${m:.2f}<br>Leverage: {l:.0f}x"
                      for i, (p, m, l) in enumerate(zip(long_trades['entry_price'], long_trades['margin'], long_trades['leverage']))],
                hoverinfo='text+x'
            ),
            row=1, col=1
        )

    # Short entries (red triangles down)
    short_trades = trades_df[trades_df['direction'] == 'Short']
    if not short_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=short_trades['entry_datetime'],
                y=short_trades['entry_price'],
                mode='markers',
                name='Short Entry',
                marker=dict(symbol='triangle-down', size=14, color='#ff4444', line=dict(width=2, color='darkred')),
                text=[f"SHORT ENTRY #{i}<br>Price: ${p:.2f}<br>Margin: ${m:.2f}<br>Leverage: {l:.0f}x"
                      for i, (p, m, l) in enumerate(zip(short_trades['entry_price'], short_trades['margin'], short_trades['leverage']))],
                hoverinfo='text+x'
            ),
            row=1, col=1
        )

    # Winning exits (green circles)
    if not winners.empty:
        fig.add_trace(
            go.Scatter(
                x=winners['exit_datetime'],
                y=winners['exit_price'],
                mode='markers',
                name='Win Exit',
                marker=dict(symbol='circle', size=12, color='#00cc00', line=dict(width=2, color='darkgreen')),
                text=[f"WIN EXIT<br>Exit: ${p:.2f}<br>PnL: +${pnl:.2f}<br>Reason: {r}"
                      for p, pnl, r in zip(winners['exit_price'], winners['pnl'], winners['exit_reason'])],
                hoverinfo='text+x'
            ),
            row=1, col=1
        )

    # Losing exits (red circles)
    if not losers.empty:
        fig.add_trace(
            go.Scatter(
                x=losers['exit_datetime'],
                y=losers['exit_price'],
                mode='markers',
                name='Loss Exit',
                marker=dict(symbol='circle', size=12, color='#ff4444', line=dict(width=2, color='darkred')),
                text=[f"LOSS EXIT<br>Exit: ${p:.2f}<br>PnL: ${pnl:.2f}<br>Reason: {r}"
                      for p, pnl, r in zip(losers['exit_price'], losers['pnl'], losers['exit_reason'])],
                hoverinfo='text+x'
            ),
            row=1, col=1
        )

    # Draw lines connecting entry to exit for each trade (more visible)
    for idx, trade in trades_df.iterrows():
        color = '#00aa00' if trade['pnl'] > 0 else '#cc0000'
        fig.add_trace(
            go.Scatter(
                x=[trade['entry_datetime'], trade['exit_datetime']],
                y=[trade['entry_price'], trade['exit_price']],
                mode='lines',
                line=dict(color=color, width=2),
                opacity=0.6,
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

    # Plot 2: Equity curve
    fig.add_trace(
        go.Scatter(
            x=trades_df_sorted['exit_datetime'],
            y=cumulative_pnl,
            mode='lines',
            name='Cumulative PnL',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,100,255,0.2)'
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

    # Plot 3: Drawdown
    fig.add_trace(
        go.Scatter(
            x=trades_df_sorted['exit_datetime'],
            y=-drawdown_pct,
            mode='lines',
            name='Drawdown %',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)'
        ),
        row=3, col=1
    )

    # Update layout
    leverage = trades_df['leverage'].iloc[0] if 'leverage' in trades_df.columns else 1.0
    total_pnl = trades_df['pnl'].sum()
    win_rate = len(winners) / len(trades_df) * 100 if len(trades_df) > 0 else 0

    fig.update_layout(
        title=f'Backtest Results: {len(trades_df)} trades | Win Rate: {win_rate:.1f}% | Total PnL: ${total_pnl:.2f} | Leverage: {leverage:.0f}x',
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest'
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=3, col=1)

    # Disable rangeslider for candlestick (takes up space), enable zoom buttons
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_layout(xaxis_rangebreaks=[dict(bounds=["sat", "mon"])])  # Hide weekends if relevant

    # Show the interactive plot
    fig.show()

    # Print summary
    print("\n" + "="*50)
    print("TRADE SUMMARY")
    print("="*50)
    print(f"Total trades: {len(trades_df)}")
    print(f"Long trades: {len(long_trades)}")
    print(f"Short trades: {len(short_trades)}")
    print(f"\nWinners: {len(winners)} ({win_rate:.1f}%)")
    print(f"Losers: {len(losers)} ({100-win_rate:.1f}%)")
    print(f"\nTotal PnL: ${total_pnl:.2f}")
    if len(winners) > 0:
        print(f"Average Win: ${winners['pnl'].mean():.2f}")
    if len(losers) > 0:
        print(f"Average Loss: ${losers['pnl'].mean():.2f}")
    print(f"\nMax Drawdown: {drawdown_pct.max():.2f}%")
    print(f"Leverage: {leverage:.0f}x")

    print("\nExit Reasons:")
    for reason, count in trades_df['exit_reason'].value_counts().items():
        print(f"  {reason}: {count} ({count/len(trades_df)*100:.1f}%)")


def parse_range(range_str: str) -> tuple:
    """Parse range string like '0-100' or '50-150'."""
    if not range_str:
        return None
    parts = range_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid range format: {range_str}. Use 'START-END' like '0-100'")
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(description='Visualize backtest trades interactively')
    parser.add_argument('--trades', default='data/backtest_trades.json',
                        help='Path to trades JSON file')
    parser.add_argument('--data', default='data/btc.csv',
                        help='Path to price data CSV')
    parser.add_argument('--range', dest='trade_range', default=None,
                        help='Trade range to display (e.g., "0-100" or "50-150")')
    parser.add_argument('--save', default=None,
                        help='Save to HTML file instead of displaying')
    args = parser.parse_args()

    print(f"Loading trades from: {args.trades}")
    trades = load_trades(args.trades)
    print(f"Loaded {len(trades)} trades")

    print(f"Loading price data from: {args.data}")
    prices = load_price_data(args.data)
    print(f"Loaded {len(prices)} price points")

    trade_range = parse_range(args.trade_range)

    plot_trades_interactive(trades, prices, trade_range)


if __name__ == '__main__':
    main()

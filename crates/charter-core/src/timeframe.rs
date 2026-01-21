//! Timeframe types and candle aggregation.

use crate::candle::Candle;

/// Timeframe enumeration for different chart periods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Timeframe {
    Min1,    // 1 minute (base data)
    Min3,    // 3 minutes
    Min5,    // 5 minutes
    Min30,   // 30 minutes
    Hour1,   // 1 hour
    Hour3,   // 3 hours
    Hour5,   // 5 hours
    Hour10,  // 10 hours
    Day1,    // 1 day
    Week1,   // 1 week
    Week3,   // 3 weeks
    Month1,  // 1 month
}

impl Timeframe {
    /// Returns the duration of this timeframe in seconds.
    pub fn seconds(&self) -> f64 {
        match self {
            Timeframe::Min1 => 60.0,
            Timeframe::Min3 => 60.0 * 3.0,
            Timeframe::Min5 => 60.0 * 5.0,
            Timeframe::Min30 => 60.0 * 30.0,
            Timeframe::Hour1 => 60.0 * 60.0,
            Timeframe::Hour3 => 60.0 * 60.0 * 3.0,
            Timeframe::Hour5 => 60.0 * 60.0 * 5.0,
            Timeframe::Hour10 => 60.0 * 60.0 * 10.0,
            Timeframe::Day1 => 60.0 * 60.0 * 24.0,
            Timeframe::Week1 => 60.0 * 60.0 * 24.0 * 7.0,
            Timeframe::Week3 => 60.0 * 60.0 * 24.0 * 21.0,
            Timeframe::Month1 => 60.0 * 60.0 * 24.0 * 30.0,
        }
    }

    /// Returns a short label for this timeframe.
    pub fn label(&self) -> &'static str {
        match self {
            Timeframe::Min1 => "1m",
            Timeframe::Min3 => "3m",
            Timeframe::Min5 => "5m",
            Timeframe::Min30 => "30m",
            Timeframe::Hour1 => "1h",
            Timeframe::Hour3 => "3h",
            Timeframe::Hour5 => "5h",
            Timeframe::Hour10 => "10h",
            Timeframe::Day1 => "1d",
            Timeframe::Week1 => "1w",
            Timeframe::Week3 => "3w",
            Timeframe::Month1 => "1M",
        }
    }

    /// Returns all available timeframes in order.
    pub fn all() -> &'static [Timeframe] {
        &[
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
            Timeframe::Week3,
            Timeframe::Month1,
        ]
    }
}

/// Aggregate candles into a larger timeframe.
pub fn aggregate_candles(candles: &[Candle], timeframe: Timeframe) -> Vec<Candle> {
    if candles.is_empty() {
        return Vec::new();
    }

    let interval = timeframe.seconds();
    let mut aggregated = Vec::new();
    let mut current_bucket: Option<Candle> = None;
    let mut current_bucket_start = 0.0;

    for candle in candles {
        let bucket_start = (candle.timestamp / interval).floor() * interval;

        if let Some(ref mut agg) = current_bucket {
            if bucket_start == current_bucket_start {
                // Same bucket - update high, low, close, accumulate volume
                agg.high = agg.high.max(candle.high);
                agg.low = agg.low.min(candle.low);
                agg.close = candle.close;
                agg.volume += candle.volume;
            } else {
                // New bucket - save current and start new
                aggregated.push(*agg);
                current_bucket = Some(Candle::new(
                    bucket_start,
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                ));
                current_bucket_start = bucket_start;
            }
        } else {
            // First candle
            current_bucket = Some(Candle::new(
                bucket_start,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            ));
            current_bucket_start = bucket_start;
        }
    }

    // Don't forget the last bucket
    if let Some(agg) = current_bucket {
        aggregated.push(agg);
    }

    aggregated
}

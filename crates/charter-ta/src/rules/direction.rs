//! Direction-based rules for candle analysis.

use super::{Rule, RuleContext};
use crate::types::CandleDirection;

/// Rule that matches if the current candle is bullish.
#[derive(Debug, Clone, Copy, Default)]
pub struct IsBullish;

impl Rule for IsBullish {
    type Output = ();

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        let meta = ctx.current_metadata()?;
        if meta.direction == CandleDirection::Bullish {
            Some(())
        } else {
            None
        }
    }

    fn min_candles(&self) -> usize {
        1
    }
}

/// Rule that matches if the current candle is bearish.
#[derive(Debug, Clone, Copy, Default)]
pub struct IsBearish;

impl Rule for IsBearish {
    type Output = ();

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        let meta = ctx.current_metadata()?;
        if meta.direction == CandleDirection::Bearish {
            Some(())
        } else {
            None
        }
    }

    fn min_candles(&self) -> usize {
        1
    }
}

/// Rule that matches if the current candle is a doji.
#[derive(Debug, Clone, Copy, Default)]
pub struct IsDoji;

impl Rule for IsDoji {
    type Output = ();

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        let meta = ctx.current_metadata()?;
        if meta.direction == CandleDirection::Doji {
            Some(())
        } else {
            None
        }
    }

    fn min_candles(&self) -> usize {
        1
    }
}

/// Rule that matches if there are N consecutive bullish candles ending at current.
#[derive(Debug, Clone, Copy)]
pub struct ConsecutiveBullish {
    pub count: usize,
}

impl ConsecutiveBullish {
    pub fn new(count: usize) -> Self {
        Self { count }
    }
}

/// Result of ConsecutiveBullish rule, containing the count found.
#[derive(Debug, Clone, Copy)]
pub struct ConsecutiveResult {
    /// Number of consecutive candles found (may be >= requested count).
    pub count: usize,
    /// Start index of the consecutive sequence.
    pub start_index: usize,
}

impl Rule for ConsecutiveBullish {
    type Output = ConsecutiveResult;

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        if !ctx.has_candles(self.count) {
            return None;
        }

        let mut consecutive = 0;
        let mut idx = ctx.current_index;

        loop {
            let meta = ctx.metadata.get(idx)?;
            if meta.direction != CandleDirection::Bullish {
                break;
            }
            consecutive += 1;
            if idx == 0 {
                break;
            }
            idx -= 1;
        }

        if consecutive >= self.count {
            Some(ConsecutiveResult {
                count: consecutive,
                start_index: ctx.current_index + 1 - consecutive,
            })
        } else {
            None
        }
    }

    fn min_candles(&self) -> usize {
        self.count
    }
}

/// Rule that matches if there are N consecutive bearish candles ending at current.
#[derive(Debug, Clone, Copy)]
pub struct ConsecutiveBearish {
    pub count: usize,
}

impl ConsecutiveBearish {
    pub fn new(count: usize) -> Self {
        Self { count }
    }
}

impl Rule for ConsecutiveBearish {
    type Output = ConsecutiveResult;

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        if !ctx.has_candles(self.count) {
            return None;
        }

        let mut consecutive = 0;
        let mut idx = ctx.current_index;

        loop {
            let meta = ctx.metadata.get(idx)?;
            if meta.direction != CandleDirection::Bearish {
                break;
            }
            consecutive += 1;
            if idx == 0 {
                break;
            }
            idx -= 1;
        }

        if consecutive >= self.count {
            Some(ConsecutiveResult {
                count: consecutive,
                start_index: ctx.current_index + 1 - consecutive,
            })
        } else {
            None
        }
    }

    fn min_candles(&self) -> usize {
        self.count
    }
}

/// Rule that matches if the current candle has a specific direction.
#[derive(Debug, Clone, Copy)]
pub struct HasDirection(pub CandleDirection);

impl Rule for HasDirection {
    type Output = ();

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        let meta = ctx.current_metadata()?;
        if meta.direction == self.0 {
            Some(())
        } else {
            None
        }
    }

    fn min_candles(&self) -> usize {
        1
    }
}

/// Rule that matches if the current candle has a minimum body ratio.
#[derive(Debug, Clone, Copy)]
pub struct MinBodyRatio {
    pub min_ratio: f32,
}

impl MinBodyRatio {
    pub fn new(min_ratio: f32) -> Self {
        Self { min_ratio }
    }
}

impl Rule for MinBodyRatio {
    type Output = f32;

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        let meta = ctx.current_metadata()?;
        if meta.body_ratio >= self.min_ratio {
            Some(meta.body_ratio)
        } else {
            None
        }
    }

    fn min_candles(&self) -> usize {
        1
    }
}

/// Rule that matches if the current candle direction changed from the previous.
#[derive(Debug, Clone, Copy, Default)]
pub struct DirectionChanged;

impl Rule for DirectionChanged {
    type Output = (CandleDirection, CandleDirection);

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        let current = ctx.current_metadata()?;
        let previous = ctx.metadata_at_offset(-1)?;

        // Skip if either is a doji
        if current.direction.is_doji() || previous.direction.is_doji() {
            return None;
        }

        if current.direction != previous.direction {
            Some((previous.direction, current.direction))
        } else {
            None
        }
    }

    fn min_candles(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CandleMetadata;
    use charter_core::Candle;

    fn make_candle(open: f32, high: f32, low: f32, close: f32) -> Candle {
        Candle::new(0.0, open, high, low, close, 100.0)
    }

    fn make_context<'a>(
        candles: &'a [Candle],
        metadata: &'a [CandleMetadata],
        current_index: usize,
    ) -> RuleContext<'a> {
        RuleContext::new(candles, metadata, current_index, &[], &[])
    }

    #[test]
    fn test_is_bullish() {
        let candles = vec![make_candle(100.0, 110.0, 95.0, 105.0)]; // Bullish
        let metadata: Vec<_> = candles
            .iter()
            .map(|c| CandleMetadata::from_candle(c, 0.0))
            .collect();

        let rule = IsBullish;
        let ctx = make_context(&candles, &metadata, 0);
        assert!(rule.evaluate(&ctx).is_some());
    }

    #[test]
    fn test_consecutive_bullish() {
        let candles = vec![
            make_candle(100.0, 110.0, 95.0, 105.0),   // Bullish
            make_candle(105.0, 115.0, 100.0, 110.0),  // Bullish
            make_candle(110.0, 120.0, 105.0, 115.0),  // Bullish
        ];
        let metadata: Vec<_> = candles
            .iter()
            .map(|c| CandleMetadata::from_candle(c, 0.0))
            .collect();

        let rule = ConsecutiveBullish::new(3);
        let ctx = make_context(&candles, &metadata, 2);
        let result = rule.evaluate(&ctx);
        assert!(result.is_some());
        assert_eq!(result.unwrap().count, 3);
    }

    #[test]
    fn test_direction_changed() {
        let candles = vec![
            make_candle(100.0, 110.0, 95.0, 105.0),   // Bullish
            make_candle(105.0, 110.0, 95.0, 100.0),   // Bearish
        ];
        let metadata: Vec<_> = candles
            .iter()
            .map(|c| CandleMetadata::from_candle(c, 0.0))
            .collect();

        let rule = DirectionChanged;
        let ctx = make_context(&candles, &metadata, 1);
        let result = rule.evaluate(&ctx);
        assert!(result.is_some());
        let (from, to) = result.unwrap();
        assert_eq!(from, CandleDirection::Bullish);
        assert_eq!(to, CandleDirection::Bearish);
    }
}

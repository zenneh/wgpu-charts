//! Rules engine for technical analysis.
//!
//! The rules engine provides a composable way to define and evaluate
//! conditions on candle data, ranges, and levels.

use charter_core::Candle;

use crate::types::{CandleMetadata, Level, Range};

mod combinators;
mod direction;

pub use combinators::{And, Not, Or, Then};
pub use direction::{
    ConsecutiveBearish, ConsecutiveBullish, ConsecutiveResult, DirectionChanged, HasDirection,
    IsBearish, IsBullish, IsDoji, MinBodyRatio,
};

/// Context provided to rules for evaluation.
///
/// Contains all the data needed to evaluate rules at a specific point in time.
pub struct RuleContext<'a> {
    /// All candles up to and including the current one.
    pub candles: &'a [Candle],
    /// Pre-computed metadata for each candle.
    pub metadata: &'a [CandleMetadata],
    /// The index of the current candle being evaluated.
    pub current_index: usize,
    /// All completed ranges so far.
    pub ranges: &'a [Range],
    /// All active levels.
    pub levels: &'a [Level],
}

impl<'a> RuleContext<'a> {
    /// Create a new rule context.
    pub fn new(
        candles: &'a [Candle],
        metadata: &'a [CandleMetadata],
        current_index: usize,
        ranges: &'a [Range],
        levels: &'a [Level],
    ) -> Self {
        Self {
            candles,
            metadata,
            current_index,
            ranges,
            levels,
        }
    }

    /// Get the current candle.
    #[inline]
    pub fn current_candle(&self) -> Option<&Candle> {
        self.candles.get(self.current_index)
    }

    /// Get the current candle's metadata.
    #[inline]
    pub fn current_metadata(&self) -> Option<&CandleMetadata> {
        self.metadata.get(self.current_index)
    }

    /// Get a candle by offset from current (negative = past, positive = future).
    #[inline]
    pub fn candle_at_offset(&self, offset: isize) -> Option<&Candle> {
        let index = self.current_index as isize + offset;
        if index < 0 {
            return None;
        }
        self.candles.get(index as usize)
    }

    /// Get metadata by offset from current.
    #[inline]
    pub fn metadata_at_offset(&self, offset: isize) -> Option<&CandleMetadata> {
        let index = self.current_index as isize + offset;
        if index < 0 {
            return None;
        }
        self.metadata.get(index as usize)
    }

    /// Get the most recent completed range.
    #[inline]
    pub fn last_range(&self) -> Option<&Range> {
        self.ranges.last()
    }

    /// Get the N most recent ranges.
    pub fn last_n_ranges(&self, n: usize) -> &[Range] {
        let len = self.ranges.len();
        if n >= len {
            self.ranges
        } else {
            &self.ranges[len - n..]
        }
    }

    /// Check if there are at least N candles available (including current).
    #[inline]
    pub fn has_candles(&self, n: usize) -> bool {
        self.current_index + 1 >= n
    }
}

/// Trait for rules that can be evaluated on candle data.
///
/// Rules are composable and can be combined using And, Or, Not, and Then.
pub trait Rule: Send + Sync {
    /// The output type of this rule when it matches.
    type Output;

    /// Evaluate the rule on the given context.
    ///
    /// Returns `Some(output)` if the rule matches, `None` otherwise.
    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output>;

    /// Minimum number of candles required for this rule.
    fn min_candles(&self) -> usize;

    /// Combine this rule with another using AND logic.
    fn and<R>(self, other: R) -> And<Self, R>
    where
        Self: Sized,
        R: Rule,
    {
        And(self, other)
    }

    /// Combine this rule with another using OR logic.
    fn or<R>(self, other: R) -> Or<Self, R>
    where
        Self: Sized,
        R: Rule,
    {
        Or(self, other)
    }

    /// Negate this rule.
    fn not(self) -> Not<Self>
    where
        Self: Sized,
    {
        Not(self)
    }

    /// Require this rule to be followed by another within N candles.
    fn then<R>(self, other: R, max_gap: usize) -> Then<Self, R>
    where
        Self: Sized,
        R: Rule,
    {
        Then::new(self, other, max_gap)
    }
}

/// A boxed rule for dynamic dispatch.
pub type BoxedRule<O> = Box<dyn Rule<Output = O> + Send + Sync>;

/// Result of evaluating a set of rules.
#[derive(Debug, Default)]
pub struct RuleResults<T> {
    pub matches: Vec<T>,
}

impl<T> RuleResults<T> {
    pub fn new() -> Self {
        Self { matches: Vec::new() }
    }

    pub fn push(&mut self, value: T) {
        self.matches.push(value);
    }

    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    pub fn len(&self) -> usize {
        self.matches.len()
    }
}

/// A rule that always matches, returning a unit value.
pub struct Always;

impl Rule for Always {
    type Output = ();

    fn evaluate(&self, _ctx: &RuleContext) -> Option<Self::Output> {
        Some(())
    }

    fn min_candles(&self) -> usize {
        0
    }
}

/// A rule that never matches.
pub struct Never;

impl Rule for Never {
    type Output = ();

    fn evaluate(&self, _ctx: &RuleContext) -> Option<Self::Output> {
        None
    }

    fn min_candles(&self) -> usize {
        0
    }
}

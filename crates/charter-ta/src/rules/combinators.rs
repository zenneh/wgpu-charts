//! Rule combinators for composing rules.

use super::{Rule, RuleContext};

/// AND combinator - both rules must match.
pub struct And<A, B>(pub A, pub B);

impl<A, B> Rule for And<A, B>
where
    A: Rule,
    B: Rule,
{
    type Output = (A::Output, B::Output);

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        let a = self.0.evaluate(ctx)?;
        let b = self.1.evaluate(ctx)?;
        Some((a, b))
    }

    fn min_candles(&self) -> usize {
        self.0.min_candles().max(self.1.min_candles())
    }
}

/// OR combinator - either rule must match (returns first match).
pub struct Or<A, B>(pub A, pub B);

/// Result of OR evaluation.
pub enum Either<A, B> {
    Left(A),
    Right(B),
}

impl<A, B> Rule for Or<A, B>
where
    A: Rule,
    B: Rule,
{
    type Output = Either<A::Output, B::Output>;

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        if let Some(a) = self.0.evaluate(ctx) {
            return Some(Either::Left(a));
        }
        if let Some(b) = self.1.evaluate(ctx) {
            return Some(Either::Right(b));
        }
        None
    }

    fn min_candles(&self) -> usize {
        self.0.min_candles().min(self.1.min_candles())
    }
}

/// NOT combinator - negates the rule.
pub struct Not<A>(pub A);

impl<A> Rule for Not<A>
where
    A: Rule,
{
    type Output = ();

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        if self.0.evaluate(ctx).is_none() {
            Some(())
        } else {
            None
        }
    }

    fn min_candles(&self) -> usize {
        self.0.min_candles()
    }
}

/// THEN combinator - first rule followed by second within max_gap candles.
///
/// This is useful for detecting patterns like:
/// "3 bearish candles followed by a bullish candle within 2 candles"
pub struct Then<A, B> {
    pub first: A,
    pub second: B,
    pub max_gap: usize,
}

impl<A, B> Then<A, B> {
    pub fn new(first: A, second: B, max_gap: usize) -> Self {
        Self {
            first,
            second,
            max_gap,
        }
    }
}

/// Result of THEN evaluation, including the gap between matches.
pub struct ThenResult<A, B> {
    pub first: A,
    pub second: B,
    pub gap: usize,
}

impl<A, B> Rule for Then<A, B>
where
    A: Rule + Clone,
    B: Rule,
    A::Output: Clone,
{
    type Output = ThenResult<A::Output, B::Output>;

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        // The second rule must match at the current position
        let second_result = self.second.evaluate(ctx)?;

        // Look back for the first rule match within max_gap
        for gap in 0..=self.max_gap {
            let offset = -(gap as isize) - 1;
            let first_index = ctx.current_index as isize + offset;

            if first_index < 0 {
                break;
            }

            // Create a context for the earlier position
            let earlier_ctx = RuleContext {
                candles: ctx.candles,
                metadata: ctx.metadata,
                current_index: first_index as usize,
                ranges: ctx.ranges,
                levels: ctx.levels,
            };

            if let Some(first_result) = self.first.evaluate(&earlier_ctx) {
                return Some(ThenResult {
                    first: first_result,
                    second: second_result,
                    gap,
                });
            }
        }

        None
    }

    fn min_candles(&self) -> usize {
        self.first.min_candles() + 1 // At least first + one more for second
    }
}

/// Macro for combining multiple rules with AND.
#[macro_export]
macro_rules! all_of {
    ($rule:expr) => { $rule };
    ($first:expr, $($rest:expr),+ $(,)?) => {
        $first.and($crate::all_of!($($rest),+))
    };
}

/// Macro for combining multiple rules with OR.
#[macro_export]
macro_rules! any_of {
    ($rule:expr) => { $rule };
    ($first:expr, $($rest:expr),+ $(,)?) => {
        $first.or($crate::any_of!($($rest),+))
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{Always, Never};

    #[test]
    fn test_and_combinator() {
        let rule = And(Always, Always);
        let ctx = RuleContext::new(&[], &[], 0, &[], &[]);
        assert!(rule.evaluate(&ctx).is_some());

        let rule = And(Always, Never);
        assert!(rule.evaluate(&ctx).is_none());
    }

    #[test]
    fn test_or_combinator() {
        let rule = Or(Never, Always);
        let ctx = RuleContext::new(&[], &[], 0, &[], &[]);
        assert!(rule.evaluate(&ctx).is_some());

        let rule = Or(Never, Never);
        assert!(rule.evaluate(&ctx).is_none());
    }

    #[test]
    fn test_not_combinator() {
        let rule = Not(Never);
        let ctx = RuleContext::new(&[], &[], 0, &[], &[]);
        assert!(rule.evaluate(&ctx).is_some());

        let rule = Not(Always);
        assert!(rule.evaluate(&ctx).is_none());
    }
}

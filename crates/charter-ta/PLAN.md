# Charter-TA: Technical Analysis Crate

## Architecture Overview

```
                    ┌─────────────────────────────────────────────┐
                    │              Analyzer                       │
                    │  (Orchestrates rules, tracks state)         │
                    └─────────────────┬───────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
     │  Range Detector │     │ Level Tracker  │     │ Trend Detector │
     │  (Rules Engine) │     │ (Hit/Broken)   │     │   (Future)     │
     └────────┬───────┘     └────────┬───────┘     └────────────────┘
              │                       │
              ▼                       ▼
     ┌────────────────┐     ┌────────────────┐
     │     Range      │────▶│     Level      │
     └────────────────┘     └────────────────┘
              │
              ▼
     ┌────────────────┐
     │    Candle      │  (from charter-core)
     └────────────────┘
```

## Core Concepts

### 1. Candle Direction
```rust
enum CandleDirection {
    Bullish,  // close > open
    Bearish,  // close < open
    Doji,     // close == open (within tolerance)
}
```

### 2. Range
A consecutive sequence of candles with the same direction.

```rust
struct Range {
    direction: CandleDirection,
    start_index: usize,       // Index in candle array
    end_index: usize,         // Inclusive
    candle_count: usize,
    high: f32,                // Max high in range
    low: f32,                 // Min low in range
    open: f32,                // First candle open
    close: f32,               // Last candle close
    total_volume: f32,
}
```

### 3. Level
A price level derived from a range with interaction tracking.

```rust
enum LevelType {
    Hold,        // Primary level
    GreedyHold,  // Secondary (more aggressive) level
}

enum LevelState {
    Active,      // Level is valid and being tracked
    Hit(u32),    // Wick touched but held (count of hits)
    Broken,      // Body closed through the level
}

struct Level {
    price: f32,
    level_type: LevelType,
    direction: CandleDirection,  // Direction of source range
    source_range_id: RangeId,
    state: LevelState,
    created_at_index: usize,
    hits: Vec<LevelHit>,         // History of interactions
}
```

### 4. Level Interactions

```
BEARISH LEVEL (resistance):

    ════════════════════  Level Price
         │    │
         │    │ ◄─── Wick touches = HIT (if body closes above)
    ┌────┴────┴────┐
    │   BODY       │ ◄─── If body closes below = BROKEN
    └──────────────┘

BULLISH LEVEL (support):

    ┌──────────────┐
    │   BODY       │ ◄─── If body closes above = still valid
    └────┬────┬────┘
         │    │ ◄─── Wick touches = HIT (if body closes above)
    ════════════════════  Level Price
```

## Rules Engine Design

### Rule Trait
```rust
trait Rule: Send + Sync {
    type Output;

    /// Evaluate the rule on the given context
    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output>;

    /// Minimum candles needed for this rule
    fn min_candles(&self) -> usize;
}

struct RuleContext<'a> {
    candles: &'a [Candle],
    current_index: usize,
    ranges: &'a [Range],
    levels: &'a [Level],
}
```

### Composable Rules
```rust
// AND combinator
struct And<A, B>(A, B);

// OR combinator
struct Or<A, B>(A, B);

// NOT combinator
struct Not<A>(A);

// Sequence combinator (A then B within N candles)
struct Then<A, B> {
    first: A,
    second: B,
    max_gap: usize,
}
```

### Built-in Rules

1. **Direction Rules**
   - `IsBullish` - Current candle is bullish
   - `IsBearish` - Current candle is bearish
   - `ConsecutiveBullish(n)` - N consecutive bullish candles
   - `ConsecutiveBearish(n)` - N consecutive bearish candles

2. **Range Rules**
   - `RangeComplete` - A range just completed (direction changed)
   - `RangeMinCandles(n)` - Range has at least N candles
   - `RangeBodyRatio(min, max)` - Body/range ratio within bounds

3. **Level Rules**
   - `LevelCreated` - New level was created
   - `LevelHit` - Level was hit (wick touch, body held)
   - `LevelBroken` - Level was broken (body closed through)

## Performance Considerations

### 1. Incremental Processing
```rust
impl Analyzer {
    /// Process a single new candle (streaming mode)
    fn process_candle(&mut self, candle: Candle) -> AnalysisResult;

    /// Process batch of candles
    fn process_batch(&mut self, candles: &[Candle]) -> Vec<AnalysisResult>;
}
```

### 2. Efficient Data Structures
- Use `SmallVec` for ranges (most analyses have few active ranges)
- Use arena allocation for levels to avoid fragmentation
- Bitflags for candle properties (direction, etc.)
- Pre-compute candle metadata on ingestion

### 3. Caching
```rust
struct CandleMetadata {
    direction: CandleDirection,
    body_size: f32,
    wick_upper: f32,
    wick_lower: f32,
    body_ratio: f32,  // body / total range
}
```

### 4. Memory Layout
```rust
// Cache-friendly layout for hot path
#[repr(C)]
struct CandleCompact {
    open: f32,
    high: f32,
    low: f32,
    close: f32,
    // Computed fields packed together
    flags: u8,  // direction, special patterns
}
```

## File Structure

```
charter-ta/
├── Cargo.toml
└── src/
    ├── lib.rs              # Public API
    ├── types/
    │   ├── mod.rs
    │   ├── direction.rs    # CandleDirection, helpers
    │   ├── range.rs        # Range struct, RangeBuilder
    │   └── level.rs        # Level, LevelState, LevelHit
    ├── rules/
    │   ├── mod.rs          # Rule trait, RuleContext
    │   ├── combinators.rs  # And, Or, Not, Then
    │   ├── direction.rs    # Direction-based rules
    │   ├── range.rs        # Range detection rules
    │   └── level.rs        # Level interaction rules
    ├── analyzer.rs         # Main Analyzer orchestrator
    └── metadata.rs         # CandleMetadata, precomputation
```

## Implementation Phases

### Phase 1: Core Types (Current)
- [x] CandleDirection and helpers
- [x] Range struct with construction
- [x] Level struct with state tracking
- [x] LevelHit for interaction history

### Phase 2: Rules Engine
- [ ] Rule trait definition
- [ ] RuleContext for evaluation
- [ ] Basic combinators (And, Or, Not)
- [ ] Direction rules implementation

### Phase 3: Range Detection
- [ ] RangeDetector that tracks current range
- [ ] Range completion detection
- [ ] Range-to-Level derivation logic

### Phase 4: Level Tracking
- [ ] LevelTracker with active levels
- [ ] Hit detection algorithm
- [ ] Break detection algorithm
- [ ] Level lifecycle management

### Phase 5: Analyzer Integration
- [ ] Streaming analyzer
- [ ] Batch processing
- [ ] Query API for active levels/ranges

### Phase 6: Performance Optimization
- [ ] CandleMetadata precomputation
- [ ] SmallVec for hot collections
- [ ] Benchmark suite

## Usage Example (Target API)

```rust
use charter_ta::{Analyzer, AnalyzerConfig, rules::*};

// Create analyzer with configuration
let mut analyzer = Analyzer::builder()
    .min_range_candles(2)        // Minimum candles to form a range
    .doji_threshold(0.001)       // Body ratio for doji detection
    .level_tolerance(0.0001)     // Price tolerance for hit detection
    .build();

// Process candles
for candle in candles {
    let result = analyzer.process_candle(candle);

    // Check for new ranges
    for range in result.new_ranges() {
        println!("New {} range: {} candles", range.direction, range.candle_count);
    }

    // Check for level events
    for event in result.level_events() {
        match event {
            LevelEvent::Created(level) => println!("New level at {}", level.price),
            LevelEvent::Hit { level, candle_index } => println!("Level {} hit", level.price),
            LevelEvent::Broken { level, candle_index } => println!("Level {} broken", level.price),
        }
    }
}

// Query active levels
let active_levels = analyzer.active_levels();
for level in active_levels.filter(|l| l.direction == CandleDirection::Bearish) {
    println!("Bearish level at {} (hit {} times)", level.price, level.hit_count());
}
```

## Custom Rules Example

```rust
use charter_ta::rules::{Rule, RuleContext};

// Custom rule: Strong bullish reversal
struct StrongBullishReversal {
    min_bearish_candles: usize,
    min_body_ratio: f32,
}

impl Rule for StrongBullishReversal {
    type Output = Range;

    fn evaluate(&self, ctx: &RuleContext) -> Option<Self::Output> {
        // Look for: bearish range followed by large bullish candle
        let current = ctx.current_candle()?;
        if current.direction() != CandleDirection::Bullish {
            return None;
        }

        let prev_range = ctx.previous_range()?;
        if prev_range.direction != CandleDirection::Bearish {
            return None;
        }

        if prev_range.candle_count < self.min_bearish_candles {
            return None;
        }

        if current.body_ratio() < self.min_body_ratio {
            return None;
        }

        Some(prev_range.clone())
    }

    fn min_candles(&self) -> usize {
        self.min_bearish_candles + 1
    }
}
```

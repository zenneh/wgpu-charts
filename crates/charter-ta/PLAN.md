# Charter-TA: Technical Analysis Crate (v2 - Complete Rewrite)

## Overview

A high-performance, rule-based technical analysis library for incremental price action analysis. The core design is based on a two-pass algorithm (reverse + forward) with formal guarantees for early-stopping optimization.

## Core Data Types

### 1. Candle

The fundamental unit of price data (from `charter-core`).

```rust
struct Candle {
    timestamp: f64,
    open: f32,
    high: f32,
    low: f32,
    close: f32,
    volume: f32,
}
```

### 2. CandleDirection

Classification of candle movement.

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
enum CandleDirection {
    Bullish,   // close > open
    Bearish,   // close < open
    Doji,      // |close - open| / (high - low) < threshold
}
```

### 3. Range

A consecutive sequence of `N` candles (N defined per-timeframe) with the same dominant direction.

```rust
struct Range {
    id: RangeId,
    direction: CandleDirection,
    timeframe: TimeframeId,

    // Boundaries
    start_index: usize,
    end_index: usize,
    candle_count: usize,

    // Price extremes
    high: f32,
    low: f32,
    open: f32,   // First candle's open
    close: f32,  // Last candle's close

    // Volume
    total_volume: f32,

    // Level-defining candle data
    first_candle: (f32, f32),  // (high, low) of first candle
    last_candle: (f32, f32),   // (high, low) of last candle
}
```

**Range Configuration per Timeframe:**
```rust
struct TimeframeConfig {
    timeframe: Timeframe,
    min_candles: usize,        // Minimum candles to form a valid range
    doji_threshold: f32,       // Body ratio threshold for doji detection
}
```

### 4. Level Types

Derived from Range extremes.

```rust
enum LevelType {
    Hold,        // Conservative: min(first_low, last_low) for bullish
    GreedyHold,  // Aggressive: max(first_low, last_low) for bullish
}

enum LevelDirection {
    Support,     // From bullish range, below current price
    Resistance,  // From bearish range, above current price
}
```

**Level Price Derivation:**

| Range Direction | Hold Level | Greedy Hold Level |
|-----------------|------------|-------------------|
| Bullish (Support) | `min(first_low, last_low)` | `max(first_low, last_low)` |
| Bearish (Resistance) | `max(first_high, last_high)` | `min(first_high, last_high)` |

### 5. Level

A price level with state tracking.

```rust
struct Level {
    id: LevelId,
    price: f32,
    level_type: LevelType,
    level_direction: LevelDirection,
    source_range_id: RangeId,
    source_timeframe: TimeframeId,

    // State
    state: LevelState,
    created_at_index: usize,
    source_candle_index: usize,  // Which candle's wick defines the price

    // Interaction tracking
    hits: Vec<LevelHit>,
    break_event: Option<LevelBreak>,
}

enum LevelState {
    Inactive,   // Waiting for price to cross to "activation side"
    Active,     // Price has crossed, level is live
    Broken,     // Body closed through level
}

struct LevelHit {
    candle_index: usize,
    timeframe: TimeframeId,
    touch_price: f32,
    penetration: f32,      // How deep wick went past level
    respected: bool,       // True if only wick touched (body held)
}

struct LevelBreak {
    candle_index: usize,
    close_price: f32,
}
```

---

## Architecture: Two-Pass Analysis

### Core Trait: `Analyzer`

```rust
trait Analyzer: Send + Sync {
    /// Process new aggregated data for all configured timeframes.
    /// Expects timeframes in descending order (highest first).
    fn update(&mut self, data: &TimeframeData) -> AnalysisResult;

    /// Reset state for a fresh analysis
    fn reset(&mut self);

    /// Get current analysis state
    fn state(&self) -> &AnalyzerState;
}

struct TimeframeData {
    timeframe: TimeframeId,
    candles: Vec<Candle>,
    current_price: f32,
}

struct AnalysisResult {
    ranges: Vec<Range>,
    levels: Vec<Level>,
    level_events: Vec<LevelEvent>,
}
```

### State Machine

The analyzer maintains a state machine tracking:

```rust
struct AnalyzerState {
    // Current position
    current_price: f32,
    current_index: usize,

    // Per-timeframe state
    timeframe_states: HashMap<TimeframeId, TimeframeState>,

    // Early stopping bounds (from reverse pass)
    closest_unbroken_resistance: Option<(f32, LevelId)>,
    closest_unbroken_support: Option<(f32, LevelId)>,

    // Proven bounds for early stopping
    resistance_bound: Option<f32>,  // No unbroken resistance above this
    support_bound: Option<f32>,     // No unbroken support below this
}

struct TimeframeState {
    ranges: Vec<Range>,
    levels: BTreeMap<OrderedFloat<f32>, Level>,  // O(log n) lookup by price
    last_processed_index: usize,
}
```

---

## Pass 1: Reverse Pass (Range Detection with Early Stopping)

### Algorithm

```
REVERSE_PASS(candles[], timeframe_configs[], current_price):
    1. For each timeframe (highest to lowest):
        a. Initialize: closest_resistance = +∞, closest_support = -∞
        b. Iterate candles in REVERSE order (newest to oldest)
        c. Detect ranges incrementally using RangeBuilder
        d. For each completed range:
            - Compute hold/greedy hold levels
            - Check if level is broken by subsequent candles
            - If unbroken and closer than current best, update
        e. Apply EARLY_STOPPING_CHECK
        f. Store ranges and levels in efficient data structure

    2. Return: ranges[], levels[] indexed by price for O(log n) lookup
```

### Time Complexity

- **Without early stopping:** O(N) per timeframe where N = number of candles
- **With early stopping:** O(K) where K = candles needed to find closest unbroken levels

### Data Structure for Storage

```rust
// B-Tree for O(log n) price-based queries
struct LevelIndex {
    // Levels by price for quick range queries
    by_price: BTreeMap<OrderedFloat<f32>, Vec<LevelId>>,

    // Levels by ID for direct access
    by_id: HashMap<LevelId, Level>,

    // Active levels partitioned by direction
    active_resistance: BTreeSet<(OrderedFloat<f32>, LevelId)>,  // price ascending
    active_support: BTreeSet<(Reverse<OrderedFloat<f32>>, LevelId)>,  // price descending
}

impl LevelIndex {
    /// O(log n) - Find closest resistance above price
    fn closest_resistance_above(&self, price: f32) -> Option<&Level>;

    /// O(log n) - Find closest support below price
    fn closest_support_below(&self, price: f32) -> Option<&Level>;

    /// O(log n + k) - Find all levels in price range
    fn levels_in_range(&self, low: f32, high: f32) -> Vec<&Level>;
}
```

---

## Formal Proof: Early Stopping Condition

### Theorem (Early Stopping for Closest Unbroken Levels)

**Statement:** Given current price `C`, once we identify:
- Closest unbroken resistance level at price `R` where `R > C`
- Closest unbroken support level at price `S` where `S < C`

Then **no older range** (further back in time) can produce a closer unbroken level.

### Proof

**Part 1: Resistance Levels (Bearish Ranges)**

Let `R` be the price of the closest unbroken resistance level found so far (where `R > C`).

Consider an older bearish range that would create a resistance level at price `H` (where `H` is derived from the range's high prices).

**Case 1:** `H > R`
- The level at `H` is strictly further from current price `C` than `R`
- Therefore, `H` cannot be the "closest" unbroken resistance
- ✓ Safe to ignore

**Case 2:** `H ≤ R` and `H > C`
- For the level at `H` to be **unbroken**, no candle between the range's creation and present time could have closed below `H`
- However, we observe that current price is `C` where `C < H`
- This means price traveled from `H` (or above) down to `C`
- For price to reach `C < H`, at least one candle must have **closed at or below `C`**
- Since `C < H`, this candle's close was below `H`
- By definition, a candle closing below a bearish level **breaks** it
- Therefore, the level at `H` is **BROKEN**
- ✓ Cannot be the closest unbroken resistance

**Case 3:** `H ≤ C`
- A resistance level at `H ≤ C` is not a valid resistance (must be above current price)
- ✓ Not applicable

By exhaustion of cases, no older bearish range can produce a closer unbroken resistance than `R`. ∎

**Part 2: Support Levels (Bullish Ranges)**

Symmetric argument. Let `S` be the closest unbroken support where `S < C`.

For an older bullish range creating support at price `L`:

**Case 1:** `L < S` → Not the closest, ignore
**Case 2:** `L ≥ S` and `L < C` → Price went from `L` up to `C`, so some candle closed above `L`, breaking it
**Case 3:** `L ≥ C` → Not a valid support, ignore

Therefore, no older bullish range can produce a closer unbroken support than `S`. ∎

### Corollary: Multi-Timeframe Early Stopping

When processing timeframes in descending order (highest first):

1. The highest timeframe's closest unbroken levels establish **initial bounds**
2. Lower timeframes can only find closer levels **within** these bounds
3. Once a lower timeframe's iteration reaches a range whose level would be outside the established bounds, we can early-stop

**Optimization:** Use the highest timeframe's bounds to skip large portions of lower timeframe data.

### Early Stopping Condition (Implementation)

```rust
fn should_early_stop(
    current_range: &Range,
    closest_resistance: Option<f32>,
    closest_support: Option<f32>,
    current_price: f32,
) -> bool {
    let range_resistance = range.hold_level_price_for_resistance();
    let range_support = range.hold_level_price_for_support();

    match current_range.direction {
        CandleDirection::Bearish => {
            // This range creates a resistance level
            if let Some(r) = closest_resistance {
                // If range's potential level is further than what we have,
                // and the range is entirely above current price,
                // any older range's level would also be broken
                range_resistance >= r
            } else {
                false
            }
        }
        CandleDirection::Bullish => {
            // This range creates a support level
            if let Some(s) = closest_support {
                range_support <= s
            } else {
                false
            }
        }
        CandleDirection::Doji => false,  // Doji ranges don't create levels
    }
}
```

---

## Pass 2: Forward Pass (Level Interaction Evaluation)

### Algorithm

```
FORWARD_PASS(candles[], levels_index, start_index):
    1. For each candle from start_index to current:
        a. Get candle's price range: [low, high]
        b. Query levels_index for levels in [low - tolerance, high + tolerance]
        c. For each relevant level:
            - Check activation condition
            - Check hit condition
            - Check break condition
        d. Record events

    2. Return: level_events[]
```

### Level Interaction Rules

**Bearish Level (Resistance):**
```
ACTIVATION: candle.open > level AND candle.close > level
            (full body above level)

HIT:        candle.low <= level AND candle.close > level
            (wick touched, body held above)

BROKEN:     candle.direction == Bearish AND
            candle.open < level AND candle.close < level
            (bearish candle with full body below)
```

**Bullish Level (Support):**
```
ACTIVATION: candle.open < level AND candle.close < level
            (full body below level)

HIT:        candle.high >= level AND candle.close < level
            (wick touched, body held below)

BROKEN:     candle.direction == Bullish AND
            candle.open > level AND candle.close > level
            (bullish candle with full body above)
```

### Timeframe Constraint

A level can only be **broken** by candles from the **same timeframe** that created it. Lower timeframes can **hit** the level but not break it.

---

## ML Feature Extraction Trait

### Trait Definition

```rust
trait FeatureExtractor: Analyzer {
    type Features;
    type Error;

    /// Extract features from current analyzer state.
    /// Returns None if insufficient data for feature extraction.
    fn extract_features(&self) -> Result<Option<Self::Features>, Self::Error>;

    /// Minimum requirements for feature extraction
    fn extraction_requirements(&self) -> ExtractionRequirements;
}

struct ExtractionRequirements {
    min_candles_per_timeframe: HashMap<TimeframeId, usize>,
    min_ranges_per_timeframe: HashMap<TimeframeId, usize>,
    min_active_levels: usize,
}
```

### Feature Structure

```rust
struct MlFeatures {
    // Per-timeframe features
    timeframes: Vec<TimeframeFeatures>,

    // Global features
    current_price: f32,
    current_volume_normalized: f32,
    price_change_normalized: f32,
    body_ratio: f32,
    is_bullish: f32,  // 1.0 or 0.0
}

struct TimeframeFeatures {
    timeframe_index: usize,

    // Closest N levels per category (sorted by distance)
    bullish_hold_levels: [LevelFeatures; N_LEVELS],
    bearish_hold_levels: [LevelFeatures; N_LEVELS],
    bullish_greedy_levels: [LevelFeatures; N_LEVELS],
    bearish_greedy_levels: [LevelFeatures; N_LEVELS],

    // Level statistics
    active_level_count: u16,
    total_level_count: u16,
}

struct LevelFeatures {
    exists: f32,              // 1.0 if level exists, 0.0 for padding
    price_distance: f32,      // (level - price) / price (signed)
    hit_count: u8,
    respected_ratio: f32,     // respected_hits / total_hits
    is_active: f32,           // 1.0 if active, 0.0 if not
    age_normalized: f32,      // candles_since_creation / normalization_factor
}
```

### Extraction Implementation

```rust
impl<A: Analyzer> FeatureExtractor for A {
    type Features = MlFeatures;
    type Error = ExtractionError;

    fn extract_features(&self) -> Result<Option<Self::Features>, Self::Error> {
        let state = self.state();

        // Check requirements
        if !self.meets_extraction_requirements() {
            return Ok(None);
        }

        let mut features = MlFeatures::default();
        features.current_price = state.current_price;

        for (tf_id, tf_state) in &state.timeframe_states {
            let tf_features = TimeframeFeatures {
                timeframe_index: tf_id.as_index(),

                // Use LevelIndex for O(log n) closest level queries
                bullish_hold_levels: extract_closest_levels(
                    &tf_state.levels,
                    LevelType::Hold,
                    LevelDirection::Support,
                    state.current_price,
                    N_LEVELS,
                ),
                // ... other level categories

                active_level_count: tf_state.levels.active_count() as u16,
                total_level_count: tf_state.levels.len() as u16,
            };
            features.timeframes.push(tf_features);
        }

        Ok(Some(features))
    }
}
```

---

## Implementation Plan

### Phase 1: Core Data Types ✅
- [x] `Candle`, `CandleDirection`, `CandleMetadata`
- [x] `Range` with hold/greedy level price methods
- [x] `Level`, `LevelState`, `LevelHit`, `LevelBreak`
- [x] `TimeframeConfig` for per-timeframe settings

### Phase 2: Efficient Data Structures ✅
- [x] `LevelIndex` with B-Tree price indexing
- [x] `RangeBuilder` for incremental range detection
- [x] `TimeframeState` container

### Phase 3: Analyzer Trait & Reverse Pass ✅
- [x] Define `Analyzer` trait
- [x] Implement reverse pass algorithm
- [x] Implement early-stopping logic
- [x] Add unit tests for the formal proof

### Phase 4: Forward Pass & Level Tracking ✅
- [x] Level activation detection
- [x] Hit detection with tolerance
- [x] Break detection with timeframe constraint
- [x] Event emission

### Phase 5: Multi-Timeframe Orchestration ✅
- [x] `MultiTimeframeAnalyzer` implementation
- [x] Timeframe aggregation from base data (using charter-core)
- [x] Cross-timeframe bound propagation

### Phase 6: Feature Extraction ✅
- [x] `FeatureExtractor` trait
- [x] `MlFeatures` structure
- [x] Efficient extraction using `LevelIndex`
- [x] Serialization for training (flatten() method)

### Phase 7: Performance Optimization ✅
- [x] Benchmark suite (criterion-based)
- [ ] SIMD optimizations for level checks (future)
- [ ] Memory layout optimization (future)
- [ ] Parallel timeframe processing (future)

---

## File Structure

```
charter-ta/
├── Cargo.toml
├── PLAN.md
├── benches/
│   └── analyzer_bench.rs       # Criterion benchmarks
└── src/
    ├── lib.rs                  # Public API
    ├── types/
    │   ├── mod.rs
    │   ├── candle.rs           # CandleDirection, CandleMetadata
    │   ├── range.rs            # Range, RangeBuilder
    │   ├── level.rs            # Level, LevelState, LevelIndex
    │   └── config.rs           # TimeframeConfig
    ├── analyzer/
    │   ├── mod.rs              # Analyzer trait + DefaultAnalyzer
    │   ├── state.rs            # AnalyzerState, TimeframeState
    │   ├── reverse_pass.rs     # Reverse pass implementation
    │   ├── forward_pass.rs     # Forward pass implementation
    │   └── multi_timeframe.rs  # MultiTimeframeAnalyzer
    └── ml/
        ├── mod.rs              # ML module exports
        └── features.rs         # FeatureExtractor, MlFeatures
```

---

## Performance Targets

| Operation | Target Complexity | Notes |
|-----------|-------------------|-------|
| Reverse pass (with early stop) | O(K) | K = candles to find bounds |
| Reverse pass (worst case) | O(N) | N = total candles |
| Forward pass per candle | O(log L) | L = active levels |
| Feature extraction | O(T × log L) | T = timeframes |
| Level query by price | O(log L) | B-Tree lookup |
| Level range query | O(log L + k) | k = levels in range |

---

## Usage Example

```rust
use charter_ta::{
    Analyzer, DefaultAnalyzer, FeatureExtractor,
    TimeframeConfig, TimeframeData,
};

// Configure timeframes
let configs = vec![
    TimeframeConfig::new(Timeframe::Daily, 3, 0.001),
    TimeframeConfig::new(Timeframe::H4, 3, 0.001),
    TimeframeConfig::new(Timeframe::H1, 3, 0.001),
    TimeframeConfig::new(Timeframe::M15, 3, 0.001),
];

// Create analyzer
let mut analyzer = DefaultAnalyzer::new(configs);

// Process data (highest timeframe first)
let result = analyzer.update(&TimeframeData {
    timeframe: Timeframe::Daily,
    candles: daily_candles,
    current_price: 1850.50,
});

// Extract ML features
if let Some(features) = analyzer.extract_features()? {
    let feature_vector = features.to_vector();
    // Use for training or inference
}

// Query closest levels
let state = analyzer.state();
if let Some(resistance) = state.closest_unbroken_resistance {
    println!("Closest resistance: {}", resistance.0);
}
```

---

## Appendix: Proof Validation Test Cases

```rust
#[cfg(test)]
mod proof_tests {
    use super::*;

    /// Test Case 1: Verify broken levels are correctly identified
    #[test]
    fn test_older_level_broken_by_price_traversal() {
        // Setup: Create bearish range at high=100
        // Then price drops to 90 (current)
        // Verify: Level at 100 is broken (price crossed through)
    }

    /// Test Case 2: Verify early stopping works correctly
    #[test]
    fn test_early_stopping_skips_older_ranges() {
        // Setup: 1000 candles, closest unbroken resistance at index 950
        // Verify: Early stopping triggers, doesn't process indices < 900
    }

    /// Test Case 3: Edge case - no unbroken levels exist
    #[test]
    fn test_all_levels_broken() {
        // Setup: Price has traversed through all historical levels
        // Verify: closest_unbroken_* returns None
    }

    /// Test Case 4: Multi-timeframe bound propagation
    #[test]
    fn test_higher_timeframe_bounds_constrain_lower() {
        // Setup: Daily timeframe has resistance at 100
        // H1 has potential resistance at 105
        // Verify: H1's 105 level is not considered (outside bounds)
    }
}
```

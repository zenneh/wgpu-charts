# Discovered Issues from Logic Analysis

## Summary

Analysis completed by three specialized agents found **44 issues** across the codebase:
- **Critical**: 10 issues (must fix before release)
- **Moderate**: 13 issues (fix for stability)
- **Minor**: 21 issues (UX and polish)

---

## Critical Issues (Must Fix)

### Drawing System

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| D1 | State machine invariant | `drawing/state.rs:241-263` | `InteractionState::Drawing { first_anchor: None }` is logically invalid but allowed |
| D2 | Degenerate drawings | `drawing/state.rs:249-259` | Zero-width/height drawings accepted (same start/end point) |
| D5 | Negative indices | `drawing/state.rs:137-155` | Negative candle_index wraps to usize::MAX in snap_target() |

### Event/Input System

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| E1 | Mode transition | `input_mode.rs:75-77` | No state cleanup on mode switch (dragging flags stuck) |
| E2 | Delete event wrong | `input_mode.rs:202-206` | Delete key returns `DrawingEvent::Cancelled` instead of delete |
| E5 | Division by zero | `coords.rs:346-349` | Zero height window causes inf/NaN aspect ratio |
| E6 | Stats panel offset | `coords.rs:385-389` | screen_to_ndc() ignores STATS_PANEL_WIDTH offset |

### State Module

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| S1 | State duplication | `state/mod.rs:239-241` | ViewState vs ChartRenderer have duplicate visible_start/count |
| S2 | Unchecked indexing | `state/document.rs:143,148` | current_ta() panics if current_timeframe out of bounds |
| S3 | Missing messages | `state/mod.rs` | No process_background_messages() implementation |

---

## Moderate Issues

### Drawing System
- D3: Anchor hover priority undefined with overlapping drawings (`state.rs:172-198`)
- D4: Ray hit detection uses bounding box instead of line distance (`state.rs:325-338`)
- D6: Race condition - release without press (`state.rs:268-275`)

### Event/Input System
- E3: Escape in Normal mode ambiguous - conflicts with legacy Exit (`input_mode.rs:127`)
- E4: Mode exit has no clear target mode (`types.rs:239-240`)
- E7: Ambiguous panning logic - `blocks_panning()` vs `allows_panning()` (`input_mode.rs:265`)
- E8: Mouse state not auto-tracked (`input_mode.rs:94-96`)
- E9: Unbounded event accumulation in EventBus (`bus.rs:74-81`)

### State Module
- S4: Direct indexing of timeframes Vec (`document.rs:136`)
- S5: Integer overflow in replay visible count (`snapshot.rs:142-150`)
- S6: Visible range overflow (`view.rs:73-77`)
- S7: guideline_values not synchronized (`view.rs:60`)
- S13: No mechanism to keep ViewState and ChartRenderer in sync

---

## Minor Issues (21 total)

### Drawing UX
- D7: Coordinate precision loss at large indices
- D8: Cancel doesn't document behavior
- D9: No visual feedback for degenerate drawings
- D10: Snap behavior near edge unclear
- D11: Ambiguous anchor vs body click
- D12: No feedback dragging out of bounds

### Event/Input UX
- E10: Command batch recursion not prevented
- E11: Replay toggle unclear
- E12: No unknown key feedback
- E13: Shortcut conflicts between modes
- E14: No snap feedback
- E15: CursorLeft not handled
- E16: Timeframe bounds unchecked

### State Module
- S8: Missing validation for TaComputed message
- S9: Potential borrow conflict in snapshot_for_render
- S10: Missing error context in load_ml_model
- S11: is_ready() doesn't verify data loaded
- S12: No graceful handling of empty timeframes
- S14: RenderSnapshot may become stale
- S15: Public fields expose internal state
- S16: AppState fields should use pub(crate)

---

## Recommended Fix Priority

### Phase 1: Critical Fixes (Blocking)
1. **E5, E6**: Coordinate system fixes (division by zero, stats panel offset)
2. **S2, S4**: Bounds checking for array indexing
3. **D2, D5**: Drawing validation (degenerate drawings, negative indices)
4. **E2**: Fix delete event mapping
5. **E1**: Add mode transition cleanup

### Phase 2: Stability Fixes
1. **S1, S7, S13**: Resolve state duplication between ViewState/ChartRenderer
2. **S3**: Implement process_background_messages()
3. **D4**: Fix ray hit detection geometry
4. **E7, E8, E9**: Input system stability

### Phase 3: Polish
1. Drawing UX improvements (D9-D12)
2. Event system feedback (E12-E16)
3. State module encapsulation (S15, S16)

---

## Files to Modify

### Critical Fixes
- `crates/charter/src/coords.rs` - E5, E6
- `crates/charter/src/state/document.rs` - S2, S4
- `crates/charter/src/drawing/state.rs` - D2, D5
- `crates/charter/src/input_mode.rs` - E1, E2
- `crates/charter/src/state/view.rs` - S6

### Stability Fixes
- `crates/charter/src/state/mod.rs` - S1, S3, S13
- `crates/charter/src/drawing/state.rs` - D3, D4
- `crates/charter/src/events/bus.rs` - E9

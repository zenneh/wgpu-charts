# Charter Architecture Refactoring Plan

## Executive Summary

Refactor Charter from a monolithic state-driven architecture to a clean event-based system with:
- Unidirectional data flow
- Mode-aware input handling
- Decoupled rendering
- Clear state machines for interaction

---

## Phase 1: Event System Foundation

### 1.1 Create Core Event Infrastructure

**New file: `crates/charter/src/events/mod.rs`**

```rust
pub mod events;
pub mod commands;
pub mod bus;
```

**Event Categories:**
```rust
// Raw input events (from winit)
pub enum InputEvent {
    KeyPressed(KeyCode),
    KeyReleased(KeyCode),
    MousePressed(MouseButton, Position),
    MouseReleased(MouseButton, Position),
    MouseMoved(Position),
    MouseWheel(Delta),
}

// Semantic application events
pub enum AppEvent {
    // View
    FitView,
    Pan { delta: Vec2 },
    Zoom { factor: f32, center: Vec2 },

    // Mode changes
    EnterMode(AppMode),
    ExitMode,

    // Drawing
    DrawingStarted { tool: DrawingTool, position: AnchorPoint },
    DrawingUpdated { position: AnchorPoint },
    DrawingCompleted,
    DrawingCancelled,
    DrawingDeleted(DrawingId),
    DrawingSelected(Option<DrawingId>),

    // Data
    TimeframeChanged(Timeframe),
    SymbolChanged(String),
    DataLoaded(TimeframeData),

    // Replay
    ReplayToggled,
    ReplayStep { direction: i32 },
    ReplayIndexSet(usize),

    // UI
    PanelToggled(Panel),
}

// Commands that modify state
pub enum Command {
    UpdateCamera(CameraUpdate),
    SetDrawingTool(DrawingTool),
    AddDrawing(Drawing),
    RemoveDrawing(DrawingId),
    UpdateDrawing(DrawingId, DrawingUpdate),
    SetReplayState(ReplayState),
    RequestRedraw,
}
```

### 1.2 Input Mode System

**New file: `crates/charter/src/input/mode.rs`**

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,           // Default viewing/panning
    Drawing(DrawingTool),
    Selecting,        // Selection mode for drawings
    Replay,           // Replay controls active
    SymbolPicker,     // Modal dialog open
}

pub struct InputContext {
    pub mode: InputMode,
    pub modifiers: Modifiers,
    pub mouse_position: Option<WorldPos>,
    pub is_dragging: bool,
}

impl InputContext {
    pub fn map_key(&self, key: KeyCode) -> Option<AppEvent> {
        match (self.mode, key) {
            // Global shortcuts (work in any mode)
            (_, KeyCode::Escape) => Some(self.escape_action()),

            // Mode-specific shortcuts
            (InputMode::Normal, KeyCode::KeyF) => Some(AppEvent::FitView),
            (InputMode::Normal, KeyCode::KeyH) => Some(AppEvent::EnterMode(InputMode::Drawing(DrawingTool::HorizontalRay))),
            (InputMode::Normal, KeyCode::KeyT) => Some(AppEvent::EnterMode(InputMode::Drawing(DrawingTool::Ray))),
            (InputMode::Normal, KeyCode::KeyB) => Some(AppEvent::EnterMode(InputMode::Drawing(DrawingTool::Rectangle))),
            (InputMode::Normal, KeyCode::KeyV) => Some(AppEvent::EnterMode(InputMode::Selecting)),

            (InputMode::Drawing(_), KeyCode::KeyG) => Some(AppEvent::ToggleSnap),
            (InputMode::Selecting, KeyCode::Delete) => Some(AppEvent::DrawingDeleted(self.selected?)),

            _ => None,
        }
    }

    fn escape_action(&self) -> AppEvent {
        match self.mode {
            InputMode::SymbolPicker => AppEvent::PanelToggled(Panel::SymbolPicker),
            InputMode::Drawing(_) => AppEvent::DrawingCancelled,
            InputMode::Selecting => AppEvent::ExitMode,
            InputMode::Replay => AppEvent::ReplayToggled,
            InputMode::Normal => AppEvent::ExitMode, // no-op or app quit
        }
    }
}
```

### 1.3 Event Bus

**New file: `crates/charter/src/events/bus.rs`**

```rust
use std::collections::VecDeque;

pub struct EventBus {
    events: VecDeque<AppEvent>,
    commands: VecDeque<Command>,
}

impl EventBus {
    pub fn emit(&mut self, event: AppEvent) {
        self.events.push_back(event);
    }

    pub fn dispatch(&mut self, cmd: Command) {
        self.commands.push_back(cmd);
    }

    pub fn drain_events(&mut self) -> impl Iterator<Item = AppEvent> + '_ {
        self.events.drain(..)
    }

    pub fn drain_commands(&mut self) -> impl Iterator<Item = Command> + '_ {
        self.commands.drain(..)
    }
}
```

---

## Phase 2: State Decomposition

### 2.1 Split Monolithic State

**Current:** Single `State` struct (3568 lines)

**Target:** Multiple focused state structs

```rust
// crates/charter/src/state/mod.rs
pub struct AppState {
    pub graphics: GraphicsState,
    pub document: DocumentState,
    pub view: ViewState,
    pub interaction: InteractionState,
    pub ui: UiState,
}

// crates/charter/src/state/graphics.rs
pub struct GraphicsState {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub renderer: ChartRenderer,
}

// crates/charter/src/state/document.rs
pub struct DocumentState {
    pub timeframes: Vec<TimeframeData>,
    pub current_timeframe: usize,
    pub drawings: DrawingStore,
    pub ta_data: Vec<TimeframeTaData>,
    pub indicators: IndicatorRegistry,
}

// crates/charter/src/state/view.rs
pub struct ViewState {
    pub camera: Camera,
    pub visible_range: Range<usize>,
    pub hover: Option<HoverState>,
}

// crates/charter/src/state/interaction.rs
pub struct InteractionState {
    pub mode: InputMode,
    pub drawing: DrawingInteraction,
    pub replay: ReplayState,
    pub selection: SelectionState,
}

// crates/charter/src/state/ui.rs
pub struct UiState {
    pub egui_ctx: egui::Context,
    pub panels: PanelState,
    pub symbol_picker: SymbolPickerState,
}
```

### 2.2 Centralized Coordinate System

**New file: `crates/charter/src/coords.rs`**

```rust
/// Screen coordinates (pixels from top-left)
#[derive(Clone, Copy)]
pub struct ScreenPos {
    pub x: f32,
    pub y: f32,
}

/// Normalized Device Coordinates (-1 to 1)
#[derive(Clone, Copy)]
pub struct NdcPos {
    pub x: f32,
    pub y: f32,
}

/// World/Chart coordinates (candle index, price)
#[derive(Clone, Copy)]
pub struct WorldPos {
    pub candle_index: f32,
    pub price: f32,
}

pub struct CoordinateSystem {
    pub screen_size: (u32, u32),
    pub camera: Camera,
    pub chart_bounds: ChartBounds,
}

impl CoordinateSystem {
    pub fn screen_to_world(&self, screen: ScreenPos) -> WorldPos {
        let ndc = self.screen_to_ndc(screen);
        self.ndc_to_world(ndc)
    }

    pub fn world_to_screen(&self, world: WorldPos) -> ScreenPos {
        let ndc = self.world_to_ndc(world);
        self.ndc_to_screen(ndc)
    }

    pub fn screen_to_ndc(&self, screen: ScreenPos) -> NdcPos {
        NdcPos {
            x: (screen.x / self.screen_size.0 as f32) * 2.0 - 1.0,
            y: 1.0 - (screen.y / self.screen_size.1 as f32) * 2.0,
        }
    }

    pub fn ndc_to_world(&self, ndc: NdcPos) -> WorldPos {
        WorldPos {
            candle_index: (ndc.x - self.camera.position.x) / self.camera.scale.x,
            price: (ndc.y - self.camera.position.y) / self.camera.scale.y,
        }
    }

    // ... other conversions
}
```

---

## Phase 3: Refactored Event Loop

### 3.1 Main Loop Structure

```rust
// crates/charter/src/app.rs
fn run_event_loop(mut state: AppState, event_loop: EventLoop<()>) {
    let mut event_bus = EventBus::new();
    let mut input_ctx = InputContext::new();

    event_loop.run(move |event, target| {
        match event {
            Event::WindowEvent { event, .. } => {
                // 1. Let egui consume if it wants
                if state.ui.egui_ctx.wants_input() {
                    handle_egui_event(&mut state.ui, &event);
                    return;
                }

                // 2. Convert to semantic events based on mode
                if let Some(app_event) = input_ctx.process_window_event(&event) {
                    event_bus.emit(app_event);
                }
            }

            Event::UserEvent(msg) => {
                // Background thread messages
                handle_background_message(&mut state, msg, &mut event_bus);
            }

            Event::AboutToWait => {
                // 3. Process all pending events
                for event in event_bus.drain_events() {
                    process_event(&mut state, event, &mut event_bus);
                }

                // 4. Execute all pending commands
                for cmd in event_bus.drain_commands() {
                    execute_command(&mut state, cmd);
                }

                // 5. Render
                if state.needs_redraw {
                    render(&mut state);
                    state.needs_redraw = false;
                }
            }

            _ => {}
        }
    });
}
```

### 3.2 Event Processing

```rust
fn process_event(state: &mut AppState, event: AppEvent, bus: &mut EventBus) {
    match event {
        AppEvent::FitView => {
            let bounds = state.document.calculate_bounds();
            bus.dispatch(Command::UpdateCamera(CameraUpdate::FitTo(bounds)));
            bus.dispatch(Command::RequestRedraw);
        }

        AppEvent::EnterMode(mode) => {
            state.interaction.mode = mode;
            bus.dispatch(Command::RequestRedraw);
        }

        AppEvent::DrawingStarted { tool, position } => {
            state.interaction.drawing.start(tool, position);
            bus.dispatch(Command::RequestRedraw);
        }

        AppEvent::DrawingCompleted => {
            if let Some(drawing) = state.interaction.drawing.complete() {
                bus.dispatch(Command::AddDrawing(drawing));
            }
            bus.emit(AppEvent::ExitMode);
        }

        // ... etc
    }
}
```

---

## Phase 4: Rendering Decoupling

### 4.1 Render State Snapshot

```rust
// Immutable snapshot of state for rendering
pub struct RenderSnapshot {
    pub camera: Camera,
    pub visible_range: Range<usize>,
    pub candles: &[CandleGpu],
    pub drawings: DrawingRenderData,
    pub ta_enabled: bool,
    pub ta_data: Option<&TaRenderData>,
    pub replay_index: Option<usize>,
}

impl AppState {
    pub fn snapshot_for_render(&self) -> RenderSnapshot {
        RenderSnapshot {
            camera: self.view.camera.clone(),
            visible_range: self.view.visible_range.clone(),
            candles: self.document.current_candles(),
            drawings: self.document.drawings.prepare_render_data(&self.interaction),
            ta_enabled: self.ui.panels.ta_enabled,
            ta_data: self.ui.panels.ta_enabled.then(|| self.document.current_ta()),
            replay_index: self.interaction.replay.current_index(),
        }
    }
}

fn render(state: &mut AppState) {
    let snapshot = state.snapshot_for_render();

    // Build egui frame
    let ui_output = build_ui(&mut state.ui, &snapshot);

    // Encode GPU commands
    let mut encoder = state.graphics.device.create_command_encoder(...);
    state.graphics.renderer.render(&mut encoder, &snapshot);
    state.graphics.renderer.render_egui(&mut encoder, ui_output);

    // Submit
    state.graphics.queue.submit(std::iter::once(encoder.finish()));
    state.graphics.surface.get_current_texture()?.present();
}
```

---

## Phase 5: Drawing System Improvements

### 5.1 Drawing Store

```rust
pub struct DrawingStore {
    drawings: Vec<Drawing>,
    id_counter: u32,
    spatial_index: SpatialIndex, // For hit testing
}

impl DrawingStore {
    pub fn add(&mut self, drawing: Drawing) -> DrawingId {
        let id = DrawingId(self.id_counter);
        self.id_counter += 1;
        self.spatial_index.insert(id, &drawing);
        self.drawings.push(drawing);
        id
    }

    pub fn hit_test(&self, pos: WorldPos, tolerance: f32) -> Option<DrawingId> {
        self.spatial_index.query(pos, tolerance)
    }

    pub fn prepare_render_data(&self, interaction: &InteractionState) -> DrawingRenderData {
        // Convert to GPU format with preview and selection state
    }
}
```

### 5.2 Drawing Interaction State Machine

```rust
pub enum DrawingInteraction {
    Idle,

    Placing {
        tool: DrawingTool,
        first_point: Option<AnchorPoint>,
        preview_point: AnchorPoint,
    },

    Editing {
        drawing_id: DrawingId,
        drag_state: DragState,
    },
}

pub enum DragState {
    None,
    DraggingAnchor { index: usize, start: AnchorPoint },
    DraggingWhole { start: WorldPos },
}

impl DrawingInteraction {
    pub fn handle_press(&mut self, pos: AnchorPoint, store: &DrawingStore) -> Option<AppEvent> {
        match self {
            Self::Idle => None,

            Self::Placing { tool, first_point, .. } => {
                match (tool, first_point) {
                    (DrawingTool::HorizontalRay, _) => {
                        // Single-click tool - complete immediately
                        Some(AppEvent::DrawingCompleted)
                    }
                    (_, None) => {
                        // First point of two-point tool
                        *first_point = Some(pos);
                        None
                    }
                    (_, Some(_)) => {
                        // Second point - complete
                        Some(AppEvent::DrawingCompleted)
                    }
                }
            }

            Self::Editing { drawing_id, drag_state } => {
                // Check for anchor hit, start drag
            }
        }
    }
}
```

---

## Implementation Order

### Sprint 1: Foundation (rust-architect)
1. Create `events/` module with event types
2. Create `coords.rs` coordinate system
3. Create `input/mode.rs` input mode system

### Sprint 2: State Decomposition (rust-feature-implementer)
1. Split State into sub-structs
2. Update all references
3. Implement RenderSnapshot

### Sprint 3: Event Loop Refactor (rust-feature-implementer)
1. Implement EventBus
2. Refactor app.rs event loop
3. Wire up event processing

### Sprint 4: Drawing System (rust-wgpu-optimizer + rust-feature-implementer)
1. Implement DrawingStore with spatial index
2. Refactor DrawingInteraction state machine
3. Optimize GPU buffer management

### Sprint 5: Testing & Polish (rust-logic-analyzer)
1. Review all control flow for logic bugs
2. Check UX edge cases
3. Verify coordinate transformations

---

## Files to Create

- `crates/charter/src/events/mod.rs`
- `crates/charter/src/events/types.rs`
- `crates/charter/src/events/bus.rs`
- `crates/charter/src/coords.rs`
- `crates/charter/src/input/mode.rs`
- `crates/charter/src/state/mod.rs` (refactor existing)
- `crates/charter/src/state/graphics.rs`
- `crates/charter/src/state/document.rs`
- `crates/charter/src/state/view.rs`
- `crates/charter/src/state/interaction.rs`
- `crates/charter/src/state/ui.rs`

## Files to Modify

- `crates/charter/src/main.rs`
- `crates/charter/src/app.rs`
- `crates/charter/src/state.rs` (split and restructure)
- `crates/charter/src/input.rs`
- `crates/charter/src/drawing/mod.rs`
- `crates/charter/src/drawing/state.rs`
- `crates/charter-render/src/lib.rs`
- `crates/charter-render/src/pipeline/drawing.rs`

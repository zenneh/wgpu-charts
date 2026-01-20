//! Charter - GPU-accelerated candlestick chart viewer.

mod app;
mod coords;
mod drawing;
mod events;
mod indicators;
mod input;
mod input_mode;
mod replay;
mod state;
// mod state_legacy; // Migrated to new state module
mod ui;

use anyhow::Result;
use winit::event_loop::EventLoop;

use app::App;

fn run() -> Result<()> {
    env_logger::init();

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
    }
}

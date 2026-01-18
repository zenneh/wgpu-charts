//! Charter - GPU-accelerated candlestick chart viewer.

mod app;
mod indicators;
mod input;
mod replay;
mod state;
mod ui;

use anyhow::Result;
use winit::event_loop::EventLoop;

use app::App;

fn run() -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }

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

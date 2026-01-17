//! Data source trait definition.

use charter_core::Candle;

/// Trait for types that can load candle data.
///
/// This trait uses `anyhow::Result` for flexible error handling.
pub trait DataSource {
    fn load(&self) -> anyhow::Result<Vec<Candle>>;
}

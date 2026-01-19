//! Live data management for real-time WebSocket updates.

use charter_core::Candle;
use mexc_api::{
    types::WsEvent,
    SpotWebSocket, SubscriptionBuilder,
};
use tokio::sync::mpsc;

use crate::mexc::ws_kline_to_candle;

/// Events emitted by the live data manager.
#[derive(Debug, Clone)]
pub enum LiveDataEvent {
    /// A candle has been updated.
    CandleUpdate {
        /// The updated candle data.
        candle: Candle,
        /// Whether the candle is closed (complete) or still forming.
        is_closed: bool,
    },
    /// WebSocket connected successfully.
    Connected,
    /// WebSocket disconnected.
    Disconnected,
    /// An error occurred.
    Error(String),
}

/// Manages live WebSocket data streams from MEXC.
pub struct LiveDataManager {
    symbol: String,
    ws: Option<SpotWebSocket>,
}

impl LiveDataManager {
    /// Create a new live data manager.
    pub fn new() -> Self {
        Self {
            symbol: String::new(),
            ws: None,
        }
    }

    /// Subscribe to live kline updates for a symbol.
    ///
    /// Returns a receiver for live data events.
    pub async fn subscribe(
        &mut self,
        symbol: &str,
    ) -> anyhow::Result<mpsc::Receiver<LiveDataEvent>> {
        // Close existing connection if any
        if let Some(ws) = self.ws.take() {
            drop(ws);
        }

        self.symbol = symbol.to_uppercase();

        let (event_tx, event_rx) = mpsc::channel(100);

        // Create WebSocket connection
        let mut ws = SpotWebSocket::new();
        let ws_rx = ws.connect().await?;

        // Subscribe to 1-minute klines
        let channels = SubscriptionBuilder::new()
            .klines(&self.symbol, "1m")
            .build();
        ws.subscribe(&channels).await?;

        // Send connected event
        let _ = event_tx.send(LiveDataEvent::Connected).await;

        // Spawn task to process WebSocket events
        let symbol = self.symbol.clone();
        tokio::spawn(async move {
            process_ws_events(ws_rx, event_tx, symbol).await;
        });

        self.ws = Some(ws);

        Ok(event_rx)
    }

    /// Unsubscribe and close the connection.
    pub async fn unsubscribe(&mut self) -> anyhow::Result<()> {
        if let Some(mut ws) = self.ws.take() {
            ws.close().await?;
        }
        Ok(())
    }

    /// Get the currently subscribed symbol.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }

    /// Check if currently connected.
    pub async fn is_connected(&self) -> bool {
        if let Some(ws) = &self.ws {
            ws.is_connected().await
        } else {
            false
        }
    }
}

impl Default for LiveDataManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Process WebSocket events and forward relevant ones.
async fn process_ws_events(
    mut ws_rx: mpsc::Receiver<WsEvent>,
    event_tx: mpsc::Sender<LiveDataEvent>,
    expected_symbol: String,
) {
    // Track the last kline timestamp to detect closed candles
    let mut last_kline_time: Option<i64> = None;

    while let Some(event) = ws_rx.recv().await {
        match event {
            WsEvent::Kline {
                symbol,
                kline,
                ..
            } => {
                // Only process events for our symbol
                if symbol.to_uppercase() != expected_symbol {
                    continue;
                }

                let candle = ws_kline_to_candle(&kline);
                let current_time = kline.time;

                // Determine if this is a new candle (previous one closed)
                let is_closed = match last_kline_time {
                    Some(last_time) => current_time > last_time,
                    None => false,
                };

                last_kline_time = Some(current_time);

                if event_tx
                    .send(LiveDataEvent::CandleUpdate { candle, is_closed })
                    .await
                    .is_err()
                {
                    // Receiver dropped, exit
                    break;
                }
            }
            WsEvent::Ping | WsEvent::Pong => {
                // Ignore ping/pong
            }
            WsEvent::Unknown { raw } => {
                // Log unknown messages for debugging
                if !raw.contains("ping") && !raw.contains("pong") {
                    eprintln!("Unknown WebSocket message: {}", &raw[..raw.len().min(100)]);
                }
            }
            _ => {
                // Ignore other event types
            }
        }
    }

    // Connection closed
    let _ = event_tx.send(LiveDataEvent::Disconnected).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_live_data_manager_creation() {
        let manager = LiveDataManager::new();
        assert_eq!(manager.symbol(), "");
    }
}

//! WebSocket client for real-time market data and account updates.

use futures_util::{SinkExt, StreamExt};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, Mutex};
use tokio::time::interval;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use crate::config::WsConfig;
use crate::error::{Error, Result};
use crate::proto::{self, push_data_body};
use crate::types::{
    StringDecimal, WsBookTicker, WsChannel, WsDepthData, WsEvent, WsKline, WsKlineData,
    WsMiniTicker, WsSubscription, WsTradesData,
};

/// WebSocket connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Not connected.
    Disconnected,
    /// Connecting.
    Connecting,
    /// Connected and ready.
    Connected,
    /// Reconnecting after disconnect.
    Reconnecting,
}

/// WebSocket client for streaming market data.
#[derive(Debug)]
pub struct SpotWebSocket {
    config: WsConfig,
    state: Arc<Mutex<ConnectionState>>,
    subscriptions: Arc<Mutex<HashSet<String>>>,
    command_tx: Option<mpsc::Sender<WsCommand>>,
    event_rx: Option<mpsc::Receiver<WsEvent>>,
}

/// Internal commands for the WebSocket connection.
#[derive(Debug)]
enum WsCommand {
    Subscribe(Vec<String>),
    Unsubscribe(Vec<String>),
    Close,
}

impl SpotWebSocket {
    /// Create a new WebSocket client with default configuration.
    pub fn new() -> Self {
        Self::with_config(WsConfig::default())
    }

    /// Create a new WebSocket client with custom configuration.
    pub fn with_config(config: WsConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(ConnectionState::Disconnected)),
            subscriptions: Arc::new(Mutex::new(HashSet::new())),
            command_tx: None,
            event_rx: None,
        }
    }

    /// Connect to the WebSocket server.
    ///
    /// Returns a receiver for WebSocket events.
    pub async fn connect(&mut self) -> Result<mpsc::Receiver<WsEvent>> {
        if self.command_tx.is_some() {
            // Already connected, return existing receiver
            if let Some(rx) = self.event_rx.take() {
                return Ok(rx);
            }
        }

        let (command_tx, command_rx) = mpsc::channel(100);
        let (event_tx, event_rx) = mpsc::channel(self.config.channel_buffer_size);

        self.command_tx = Some(command_tx);

        let config = self.config.clone();
        let state = Arc::clone(&self.state);
        let subscriptions = Arc::clone(&self.subscriptions);

        // Spawn the connection task
        tokio::spawn(async move {
            run_connection(config, state, subscriptions, command_rx, event_tx).await;
        });

        // Wait for connection to be established (up to 15 seconds)
        let mut attempts = 0;
        while attempts < 150 {
            let current_state = *self.state.lock().await;
            if current_state == ConnectionState::Connected {
                return Ok(event_rx);
            }
            if current_state == ConnectionState::Disconnected && attempts > 50 {
                return Err(Error::ConnectionClosed);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
            attempts += 1;
        }

        Err(Error::Timeout)
    }

    /// Subscribe to channels.
    pub async fn subscribe(&self, channels: &[WsChannel]) -> Result<()> {
        let channel_strings: Vec<String> =
            channels.iter().map(|c| c.to_channel_string()).collect();

        // Add to subscriptions set
        {
            let mut subs = self.subscriptions.lock().await;
            for ch in &channel_strings {
                subs.insert(ch.clone());
            }
        }

        // Send subscribe command
        if let Some(tx) = &self.command_tx {
            tx.send(WsCommand::Subscribe(channel_strings))
                .await
                .map_err(|e| Error::ChannelSend(e.to_string()))?;
        }

        Ok(())
    }

    /// Unsubscribe from channels.
    pub async fn unsubscribe(&self, channels: &[WsChannel]) -> Result<()> {
        let channel_strings: Vec<String> =
            channels.iter().map(|c| c.to_channel_string()).collect();

        // Remove from subscriptions set
        {
            let mut subs = self.subscriptions.lock().await;
            for ch in &channel_strings {
                subs.remove(ch);
            }
        }

        // Send unsubscribe command
        if let Some(tx) = &self.command_tx {
            tx.send(WsCommand::Unsubscribe(channel_strings))
                .await
                .map_err(|e| Error::ChannelSend(e.to_string()))?;
        }

        Ok(())
    }

    /// Close the WebSocket connection.
    pub async fn close(&mut self) -> Result<()> {
        if let Some(tx) = self.command_tx.take() {
            let _ = tx.send(WsCommand::Close).await;
        }

        *self.state.lock().await = ConnectionState::Disconnected;

        Ok(())
    }

    /// Get the current connection state.
    pub async fn state(&self) -> ConnectionState {
        *self.state.lock().await
    }

    /// Check if connected.
    pub async fn is_connected(&self) -> bool {
        *self.state.lock().await == ConnectionState::Connected
    }

    /// Get current subscriptions.
    pub async fn subscriptions(&self) -> HashSet<String> {
        self.subscriptions.lock().await.clone()
    }
}

impl Default for SpotWebSocket {
    fn default() -> Self {
        Self::new()
    }
}

/// Run the WebSocket connection loop.
async fn run_connection(
    config: WsConfig,
    state: Arc<Mutex<ConnectionState>>,
    subscriptions: Arc<Mutex<HashSet<String>>>,
    mut command_rx: mpsc::Receiver<WsCommand>,
    event_tx: mpsc::Sender<WsEvent>,
) {
    let mut reconnect_attempts = 0;

    loop {
        *state.lock().await = ConnectionState::Connecting;

        // Connect to WebSocket
        let ws_stream = match connect_async(&config.url).await {
            Ok((stream, _)) => stream,
            Err(e) => {
                tracing::warn!("WebSocket connection failed: {}", e);

                if !config.auto_reconnect
                    || reconnect_attempts >= config.max_reconnect_attempts
                {
                    *state.lock().await = ConnectionState::Disconnected;
                    return;
                }

                reconnect_attempts += 1;
                *state.lock().await = ConnectionState::Reconnecting;
                tokio::time::sleep(config.reconnect_delay * reconnect_attempts).await;
                continue;
            }
        };

        reconnect_attempts = 0;
        *state.lock().await = ConnectionState::Connected;

        let (mut write, mut read) = ws_stream.split();

        // Resubscribe to previous channels
        {
            let subs = subscriptions.lock().await;
            if !subs.is_empty() {
                let channels: Vec<String> = subs.iter().cloned().collect();
                let sub_msg = WsSubscription::subscribe(channels);
                if let Ok(json) = serde_json::to_string(&sub_msg) {
                    let _ = write.send(Message::Text(json)).await;
                }
            }
        }

        // Create ping interval
        let mut ping_interval = interval(config.ping_interval);

        loop {
            tokio::select! {
                // Handle incoming messages
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            // JSON text message (subscription acks, etc.)
                            if let Some(event) = parse_message(&text) {
                                if event_tx.send(event).await.is_err() {
                                    tracing::warn!("Event receiver dropped");
                                    return;
                                }
                            }
                        }
                        Some(Ok(Message::Binary(data))) => {
                            // Protobuf binary message (market data)
                            if let Some(event) = parse_protobuf_message(&data) {
                                if event_tx.send(event).await.is_err() {
                                    tracing::warn!("Event receiver dropped");
                                    return;
                                }
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            let _ = write.send(Message::Pong(data)).await;
                        }
                        Some(Ok(Message::Pong(_))) => {
                            // Pong received, connection is alive
                        }
                        Some(Ok(Message::Close(_))) => {
                            tracing::info!("WebSocket closed by server");
                            break;
                        }
                        Some(Err(e)) => {
                            tracing::warn!("WebSocket error: {}", e);
                            break;
                        }
                        None => {
                            tracing::info!("WebSocket stream ended");
                            break;
                        }
                        _ => {}
                    }
                }

                // Handle commands
                cmd = command_rx.recv() => {
                    match cmd {
                        Some(WsCommand::Subscribe(channels)) => {
                            let sub_msg = WsSubscription::subscribe(channels);
                            if let Ok(json) = serde_json::to_string(&sub_msg) {
                                let _ = write.send(Message::Text(json)).await;
                            }
                        }
                        Some(WsCommand::Unsubscribe(channels)) => {
                            let unsub_msg = WsSubscription::unsubscribe(channels);
                            if let Ok(json) = serde_json::to_string(&unsub_msg) {
                                let _ = write.send(Message::Text(json)).await;
                            }
                        }
                        Some(WsCommand::Close) | None => {
                            let _ = write.send(Message::Close(None)).await;
                            *state.lock().await = ConnectionState::Disconnected;
                            return;
                        }
                    }
                }

                // Send pings
                _ = ping_interval.tick() => {
                    if write.send(Message::Ping(vec![])).await.is_err() {
                        tracing::warn!("Failed to send ping");
                        break;
                    }
                }
            }
        }

        // Connection lost, try to reconnect
        if !config.auto_reconnect || reconnect_attempts >= config.max_reconnect_attempts {
            *state.lock().await = ConnectionState::Disconnected;
            return;
        }

        reconnect_attempts += 1;
        *state.lock().await = ConnectionState::Reconnecting;
        tokio::time::sleep(config.reconnect_delay * reconnect_attempts).await;
    }
}

/// Parse a WebSocket message into an event.
fn parse_message(text: &str) -> Option<WsEvent> {
    // Try to parse the message
    let value: serde_json::Value = serde_json::from_str(text).ok()?;

    // Check for ping/pong
    if value.get("ping").is_some() {
        return Some(WsEvent::Ping);
    }
    if value.get("pong").is_some() {
        return Some(WsEvent::Pong);
    }

    // Try new V3 format first (with "channel" key instead of "c")
    if let Some(channel) = value.get("channel").and_then(|c| c.as_str()) {
        let timestamp = value.get("createtime").and_then(|t| t.as_i64()).unwrap_or(0);
        let symbol = value
            .get("symbol")
            .and_then(|s| s.as_str())
            .map(|s| s.to_string())
            .unwrap_or_default();

        // Parse kline in new V3 format
        if channel.contains("kline") {
            if let Some(kline_data) = value.get("publicspotkline") {
                if let Ok(kline) = serde_json::from_value::<WsKline>(kline_data.clone()) {
                    // Extract interval from channel name (e.g., ...@Min1)
                    let interval = channel.split('@').last().unwrap_or("Min1").to_string();
                    return Some(WsEvent::Kline {
                        symbol,
                        interval,
                        kline,
                        timestamp,
                    });
                }
            }
        }

        // Return unknown for other V3 messages we don't handle yet
        return Some(WsEvent::Unknown {
            raw: text.to_string(),
        });
    }

    // Fall back to old format (with "c" key)
    let channel = value.get("c")?.as_str()?;
    let timestamp = value.get("t").and_then(|t| t.as_i64()).unwrap_or(0);
    let symbol = value
        .get("s")
        .and_then(|s| s.as_str())
        .map(|s| s.to_string());

    // Parse based on channel type
    if channel.contains("deals") {
        if let Ok(data) = serde_json::from_value::<WsTradesData>(value.get("d")?.clone()) {
            return Some(WsEvent::Trade {
                symbol: symbol.unwrap_or_default(),
                trades: data.deals,
                timestamp,
            });
        }
    } else if channel.contains("kline") {
        if let Ok(data) = serde_json::from_value::<WsKlineData>(value.get("d")?.clone()) {
            // Extract interval from channel name
            let interval = channel.split('@').last().unwrap_or("1m").to_string();
            return Some(WsEvent::Kline {
                symbol: symbol.unwrap_or_default(),
                interval,
                kline: data.kline,
                timestamp,
            });
        }
    } else if channel.contains("depth") {
        if let Ok(data) = serde_json::from_value::<WsDepthData>(value.get("d")?.clone()) {
            return Some(WsEvent::Depth {
                symbol: symbol.unwrap_or_default(),
                bids: data.bids,
                asks: data.asks,
                version: data.version,
                timestamp,
            });
        }
    } else if channel.contains("miniTicker") {
        if let Ok(ticker) = serde_json::from_value::<WsMiniTicker>(value.get("d")?.clone()) {
            return Some(WsEvent::MiniTicker { ticker, timestamp });
        }
    } else if channel.contains("bookTicker") {
        if let Ok(ticker) = serde_json::from_value::<WsBookTicker>(value.get("d")?.clone()) {
            return Some(WsEvent::BookTicker { ticker, timestamp });
        }
    }

    // Unknown message type
    Some(WsEvent::Unknown {
        raw: text.to_string(),
    })
}

/// Parse a binary protobuf WebSocket message into an event.
fn parse_protobuf_message(data: &[u8]) -> Option<WsEvent> {
    use prost::Message;

    // Try to decode the wrapper message
    let wrapper = proto::PushDataV3ApiWrapper::decode(data).ok()?;

    let channel = wrapper.channel.clone();
    let symbol = wrapper.symbol.unwrap_or_default();
    let timestamp = wrapper.create_time.unwrap_or(0);

    // Check the body type
    match wrapper.body? {
        push_data_body::Body::PublicSpotKline(kline_data) => {
            // Convert protobuf kline to WsKline
            let kline = WsKline {
                time: kline_data.window_start,
                open: StringDecimal::from_string(kline_data.opening_price),
                high: StringDecimal::from_string(kline_data.highest_price),
                low: StringDecimal::from_string(kline_data.lowest_price),
                close: StringDecimal::from_string(kline_data.closing_price),
                volume: StringDecimal::from_string(kline_data.volume),
                quote_volume: StringDecimal::from_string(kline_data.amount),
                interval: Some(kline_data.interval.clone()),
                window_end: Some(kline_data.window_end),
            };

            Some(WsEvent::Kline {
                symbol,
                interval: kline_data.interval,
                kline,
                timestamp,
            })
        }
        // Other message types not fully implemented yet
        _ => Some(WsEvent::Unknown {
            raw: format!("Protobuf channel: {}", channel),
        }),
    }
}

/// Builder for subscribing to multiple channels.
#[derive(Debug, Default)]
pub struct SubscriptionBuilder {
    channels: Vec<WsChannel>,
}

impl SubscriptionBuilder {
    /// Create a new subscription builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Subscribe to trades for a symbol.
    pub fn trades(mut self, symbol: &str) -> Self {
        self.channels.push(WsChannel::Trades(symbol.to_uppercase()));
        self
    }

    /// Subscribe to klines for a symbol.
    pub fn klines(mut self, symbol: &str, interval: &str) -> Self {
        self.channels.push(WsChannel::Klines(
            symbol.to_uppercase(),
            interval.to_string(),
        ));
        self
    }

    /// Subscribe to incremental depth for a symbol.
    pub fn depth(mut self, symbol: &str) -> Self {
        self.channels
            .push(WsChannel::DepthIncremental(symbol.to_uppercase()));
        self
    }

    /// Subscribe to limit depth for a symbol.
    pub fn depth_limit(mut self, symbol: &str, levels: u32) -> Self {
        self.channels
            .push(WsChannel::DepthLimit(symbol.to_uppercase(), levels));
        self
    }

    /// Subscribe to mini ticker for a symbol.
    pub fn mini_ticker(mut self, symbol: &str) -> Self {
        self.channels
            .push(WsChannel::MiniTicker(symbol.to_uppercase()));
        self
    }

    /// Subscribe to all mini tickers.
    pub fn all_mini_tickers(mut self) -> Self {
        self.channels.push(WsChannel::AllMiniTickers);
        self
    }

    /// Subscribe to book ticker for a symbol.
    pub fn book_ticker(mut self, symbol: &str) -> Self {
        self.channels
            .push(WsChannel::BookTicker(symbol.to_uppercase()));
        self
    }

    /// Build the channel list.
    pub fn build(self) -> Vec<WsChannel> {
        self.channels
    }
}

/// WebSocket client for private/authenticated streams.
#[derive(Debug)]
pub struct PrivateWebSocket {
    #[allow(dead_code)]
    config: WsConfig,
    listen_key: String,
    inner: SpotWebSocket,
}

impl PrivateWebSocket {
    /// Create a new private WebSocket client.
    pub fn new(listen_key: impl Into<String>) -> Self {
        let mut config = WsConfig::default();
        let listen_key = listen_key.into();

        // Append listen key to URL
        config.url = format!("{}?listenKey={}", config.url, listen_key);

        Self {
            config: config.clone(),
            listen_key,
            inner: SpotWebSocket::with_config(config),
        }
    }

    /// Connect to the private WebSocket stream.
    pub async fn connect(&mut self) -> Result<mpsc::Receiver<WsEvent>> {
        self.inner.connect().await
    }

    /// Subscribe to account updates.
    pub async fn subscribe_account(&self) -> Result<()> {
        self.inner.subscribe(&[WsChannel::Account]).await
    }

    /// Subscribe to order updates.
    pub async fn subscribe_orders(&self) -> Result<()> {
        self.inner.subscribe(&[WsChannel::Orders]).await
    }

    /// Subscribe to trade updates.
    pub async fn subscribe_trades(&self) -> Result<()> {
        self.inner.subscribe(&[WsChannel::Trades24h]).await
    }

    /// Close the connection.
    pub async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }

    /// Get the listen key.
    pub fn listen_key(&self) -> &str {
        &self.listen_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_strings() {
        // V3 protobuf format with .pb suffix
        assert_eq!(
            WsChannel::Trades("BTCUSDT".to_string()).to_channel_string(),
            "spot@public.deals.v3.api.pb@BTCUSDT"
        );

        // Klines channel with interval conversion (1m -> Min1)
        assert_eq!(
            WsChannel::Klines("BTCUSDT".to_string(), "1m".to_string()).to_channel_string(),
            "spot@public.kline.v3.api.pb@BTCUSDT@Min1"
        );

        assert_eq!(
            WsChannel::DepthLimit("BTCUSDT".to_string(), 20).to_channel_string(),
            "spot@public.limit.depth.v3.api.pb@BTCUSDT@20"
        );
    }

    #[test]
    fn test_interval_conversion() {
        assert_eq!(
            WsChannel::Klines("BTC".to_string(), "5m".to_string()).to_channel_string(),
            "spot@public.kline.v3.api.pb@BTC@Min5"
        );
        assert_eq!(
            WsChannel::Klines("BTC".to_string(), "1h".to_string()).to_channel_string(),
            "spot@public.kline.v3.api.pb@BTC@Min60"
        );
        assert_eq!(
            WsChannel::Klines("BTC".to_string(), "1d".to_string()).to_channel_string(),
            "spot@public.kline.v3.api.pb@BTC@Day1"
        );
    }

    #[test]
    fn test_subscription_builder() {
        let channels = SubscriptionBuilder::new()
            .trades("BTCUSDT")
            .klines("BTCUSDT", "1m")
            .depth("ETHUSDT")
            .build();

        assert_eq!(channels.len(), 3);
    }
}

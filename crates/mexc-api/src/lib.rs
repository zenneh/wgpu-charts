//! # MEXC API Client Library
//!
//! A comprehensive Rust client library for interacting with the MEXC cryptocurrency exchange API.
//!
//! ## Features
//!
//! - **Spot Trading API v3**: Full support for spot trading operations
//! - **WebSocket Streams**: Real-time market data and account updates via channels
//! - **Authentication**: HMAC-SHA256 signature generation
//! - **Rate Limiting**: Built-in rate limit handling with configurable policies
//! - **Type Safety**: Strongly typed request/response models
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use mexc_api::{MexcClient, spot::SpotApi};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), mexc_api::Error> {
//!     // Create unauthenticated client for public endpoints
//!     let client = MexcClient::public()?;
//!     let spot = SpotApi::new(client);
//!
//!     // Get server time
//!     let time = spot.market().time().await?;
//!     println!("Server time: {}", time.server_time);
//!
//!     // Get ticker price
//!     let ticker = spot.market().ticker_price_single("BTCUSDT").await?;
//!     println!("BTC/USDT price: {}", ticker.price);
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Authenticated Requests
//!
//! ```rust,ignore
//! use mexc_api::{MexcClient, spot::SpotApi};
//!
//! let client = MexcClient::with_credentials(
//!     "your-api-key",
//!     "your-api-secret",
//! )?;
//! let spot = SpotApi::new(client);
//!
//! // Get account information
//! let account = spot.account().info().await?;
//! println!("Balances: {:?}", account.balances);
//! ```
//!
//! ## WebSocket Streams
//!
//! ```rust,ignore
//! use mexc_api::websocket::{SpotWebSocket, SubscriptionBuilder};
//!
//! let mut ws = SpotWebSocket::new();
//! let mut rx = ws.connect().await?;
//!
//! // Subscribe to channels
//! let channels = SubscriptionBuilder::new()
//!     .trades("BTCUSDT")
//!     .klines("BTCUSDT", "1m")
//!     .build();
//! ws.subscribe(&channels).await?;
//!
//! // Receive events via channel
//! while let Some(event) = rx.recv().await {
//!     match event {
//!         WsEvent::Trade { symbol, trades, .. } => {
//!             println!("{} trades: {:?}", symbol, trades);
//!         }
//!         _ => {}
//!     }
//! }
//! ```
//!
//! ## Trading
//!
//! ```rust,ignore
//! use mexc_api::types::{OrderSide, OrderType};
//! use mexc_api::spot::OrderOptions;
//!
//! // Place a limit order
//! let order = spot.trading().new_order(
//!     "BTCUSDT",
//!     OrderSide::Buy,
//!     OrderType::Limit,
//!     OrderOptions::new()
//!         .quantity("0.001")
//!         .price("50000"),
//! ).await?;
//!
//! // Cancel the order
//! spot.trading().cancel_order("BTCUSDT", &order.order_id).await?;
//! ```
//!
//! ## Configuration
//!
//! ```rust,ignore
//! use mexc_api::{Config, RateLimitConfig, MexcClient};
//! use std::time::Duration;
//!
//! let config = Config::new("api-key", "api-secret")
//!     .with_timeout(Duration::from_secs(30))
//!     .with_recv_window(5000)
//!     .with_rate_limit_config(RateLimitConfig::conservative());
//!
//! let client = MexcClient::new(config)?;
//! ```
//!
//! ## Documentation
//!
//! For comprehensive API documentation, see `documentation.md` in this crate.

#![warn(missing_docs)]

pub mod client;
pub mod config;
pub mod error;
pub mod proto;
pub mod rate_limit;
pub mod spot;
pub mod types;
pub mod websocket;

// Re-exports for convenience
pub use client::MexcClient;
pub use config::{Config, RateLimitConfig, WsConfig};
pub use error::{ApiError, Error, Result};
pub use spot::SpotApi;
pub use websocket::{PrivateWebSocket, SpotWebSocket, SubscriptionBuilder};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Base URL for MEXC Spot API
pub const SPOT_BASE_URL: &str = "https://api.mexc.com";

/// Base URL for MEXC Futures API
pub const FUTURES_BASE_URL: &str = "https://contract.mexc.com";

/// WebSocket URL for MEXC Spot streams
pub const SPOT_WS_URL: &str = "wss://wbs-api.mexc.com/ws";

/// WebSocket URL for MEXC Futures streams
pub const FUTURES_WS_URL: &str = "wss://contract.mexc.com/edge";

/// Prelude module for convenient imports.
pub mod prelude {
    //! Common imports for using the MEXC API client.

    pub use crate::client::MexcClient;
    pub use crate::config::{Config, RateLimitConfig, WsConfig};
    pub use crate::error::{Error, Result};
    pub use crate::spot::{AccountApi, MarketApi, OrderOptions, SpotApi, TradingApi};
    pub use crate::types::{
        AccountInfo, Balance, BookTicker, Kline, KlineInterval, NewOrderFull, Order, OrderBook,
        OrderSide, OrderStatus, OrderType, PriceTicker, ServerTime, Ticker24hr, TimeInForce,
        Trade, WsChannel, WsEvent,
    };
    pub use crate::websocket::{PrivateWebSocket, SpotWebSocket, SubscriptionBuilder};
}

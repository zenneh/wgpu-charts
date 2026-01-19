//! Integration tests for MEXC API client.
//!
//! Note: These tests require network access and may be rate-limited.
//! Tests marked with `#[ignore]` require API credentials.

use mexc_api::prelude::*;
use std::time::Duration;

/// Test creating a public client.
#[test]
fn test_create_public_client() {
    let client = MexcClient::public();
    assert!(client.is_ok());
}

/// Test creating a client with credentials.
#[test]
fn test_create_authenticated_client() {
    let client = MexcClient::with_credentials("test_key", "test_secret");
    assert!(client.is_ok());

    let client = client.unwrap();
    assert!(client.config().has_credentials());
}

/// Test configuration builder.
#[test]
fn test_config_builder() {
    let config = Config::new("api_key", "api_secret")
        .with_timeout(Duration::from_secs(60))
        .with_recv_window(10000)
        .with_rate_limiting(true)
        .with_rate_limit_config(RateLimitConfig::conservative());

    assert_eq!(config.timeout, Duration::from_secs(60));
    assert_eq!(config.recv_window, 10000);
    assert!(config.rate_limiting);
    assert!(config.has_credentials());
}

/// Test rate limit configurations.
#[test]
fn test_rate_limit_configs() {
    let default = RateLimitConfig::default();
    assert_eq!(default.ip_limit, 500);
    assert_eq!(default.ip_window, Duration::from_secs(10));

    let conservative = RateLimitConfig::conservative();
    assert_eq!(conservative.ip_limit, 400);

    let aggressive = RateLimitConfig::aggressive();
    assert_eq!(aggressive.ip_limit, 480);

    let disabled = RateLimitConfig::disabled();
    assert_eq!(disabled.ip_limit, u32::MAX);
}

/// Test WebSocket configuration.
#[test]
fn test_ws_config() {
    let config = WsConfig::default()
        .with_url("wss://custom.example.com/ws")
        .with_ping_interval(Duration::from_secs(60))
        .with_auto_reconnect(false)
        .with_buffer_size(500);

    assert_eq!(config.url, "wss://custom.example.com/ws");
    assert_eq!(config.ping_interval, Duration::from_secs(60));
    assert!(!config.auto_reconnect);
    assert_eq!(config.channel_buffer_size, 500);
}

mod types {
    use mexc_api::types::*;
    use rust_decimal::Decimal;
    use std::str::FromStr;

    /// Test StringDecimal deserialization from string.
    #[test]
    fn test_string_decimal_from_string() {
        let json = r#""123.456""#;
        let decimal: StringDecimal = serde_json::from_str(json).unwrap();
        assert_eq!(decimal.0, Decimal::from_str("123.456").unwrap());
    }

    /// Test StringDecimal deserialization from number.
    #[test]
    fn test_string_decimal_from_number() {
        let json = r#"123.456"#;
        let decimal: StringDecimal = serde_json::from_str(json).unwrap();
        // Note: floating point may have precision issues
        assert!(decimal.0 > Decimal::from_str("123.45").unwrap());
        assert!(decimal.0 < Decimal::from_str("123.46").unwrap());
    }

    /// Test OrderSide serialization.
    #[test]
    fn test_order_side() {
        assert_eq!(OrderSide::Buy.to_string(), "BUY");
        assert_eq!(OrderSide::Sell.to_string(), "SELL");

        let buy: OrderSide = serde_json::from_str(r#""BUY""#).unwrap();
        assert_eq!(buy, OrderSide::Buy);

        let sell: OrderSide = serde_json::from_str(r#""SELL""#).unwrap();
        assert_eq!(sell, OrderSide::Sell);
    }

    /// Test OrderType serialization.
    #[test]
    fn test_order_type() {
        assert_eq!(OrderType::Limit.to_string(), "LIMIT");
        assert_eq!(OrderType::Market.to_string(), "MARKET");
        assert_eq!(OrderType::LimitMaker.to_string(), "LIMIT_MAKER");

        let limit: OrderType = serde_json::from_str(r#""LIMIT""#).unwrap();
        assert_eq!(limit, OrderType::Limit);
    }

    /// Test OrderStatus deserialization.
    #[test]
    fn test_order_status() {
        let new: OrderStatus = serde_json::from_str(r#""NEW""#).unwrap();
        assert_eq!(new, OrderStatus::New);

        let filled: OrderStatus = serde_json::from_str(r#""FILLED""#).unwrap();
        assert_eq!(filled, OrderStatus::Filled);

        let canceled: OrderStatus = serde_json::from_str(r#""CANCELED""#).unwrap();
        assert_eq!(canceled, OrderStatus::Canceled);
    }

    /// Test KlineInterval serialization.
    #[test]
    fn test_kline_interval() {
        assert_eq!(KlineInterval::OneMinute.to_string(), "1m");
        assert_eq!(KlineInterval::FiveMinutes.to_string(), "5m");
        assert_eq!(KlineInterval::OneHour.to_string(), "1h");
        assert_eq!(KlineInterval::OneDay.to_string(), "1d");
        assert_eq!(KlineInterval::OneMonth.to_string(), "1M");
    }

    /// Test TimeInForce serialization.
    #[test]
    fn test_time_in_force() {
        assert_eq!(TimeInForce::Gtc.to_string(), "GTC");
        assert_eq!(TimeInForce::Ioc.to_string(), "IOC");
        assert_eq!(TimeInForce::Fok.to_string(), "FOK");
    }

    /// Test Kline deserialization from array.
    #[test]
    fn test_kline_deserialization() {
        let json = r#"[
            1704067200000,
            "50000.00",
            "50100.00",
            "49900.00",
            "50050.00",
            "100.5",
            1704070800000,
            "5025000.00",
            150,
            "60.5",
            "3030000.00",
            "0"
        ]"#;

        let kline: Kline = serde_json::from_str(json).unwrap();
        assert_eq!(kline.open_time, 1704067200000);
        assert_eq!(kline.close_time, 1704070800000);
        assert_eq!(kline.trades, 150);
    }

    /// Test Balance total calculation.
    #[test]
    fn test_balance_total() {
        let balance = Balance {
            asset: "BTC".to_string(),
            free: StringDecimal::new(Decimal::from_str("1.5").unwrap()),
            locked: StringDecimal::new(Decimal::from_str("0.5").unwrap()),
        };

        assert_eq!(balance.total(), Decimal::from_str("2.0").unwrap());
    }

    /// Test WsChannel string generation.
    #[test]
    fn test_ws_channel_strings() {
        assert_eq!(
            WsChannel::Trades("BTCUSDT".to_string()).to_channel_string(),
            "spot@public.deals.v3.api@BTCUSDT"
        );

        assert_eq!(
            WsChannel::Klines("BTCUSDT".to_string(), "1m".to_string()).to_channel_string(),
            "spot@public.kline.v3.api@BTCUSDT@1m"
        );

        assert_eq!(
            WsChannel::DepthIncremental("BTCUSDT".to_string()).to_channel_string(),
            "spot@public.increase.depth.v3.api@BTCUSDT"
        );

        assert_eq!(
            WsChannel::DepthLimit("BTCUSDT".to_string(), 20).to_channel_string(),
            "spot@public.limit.depth.v3.api@BTCUSDT@20"
        );

        assert_eq!(
            WsChannel::MiniTicker("BTCUSDT".to_string()).to_channel_string(),
            "spot@public.miniTicker.v3.api@BTCUSDT"
        );

        assert_eq!(
            WsChannel::AllMiniTickers.to_channel_string(),
            "spot@public.miniTickers.v3.api"
        );
    }
}

mod error {
    use mexc_api::error::{ApiError, SpotErrorCode};

    /// Test API error creation.
    #[test]
    fn test_api_error() {
        let err = ApiError::new(30014, "Invalid symbol");
        assert_eq!(err.code, 30014);
        assert_eq!(err.message, "Invalid symbol");
        assert!(err.is_param_error());
    }

    /// Test rate limit error detection.
    #[test]
    fn test_rate_limit_error() {
        let err = ApiError::new(429, "Too many requests");
        assert!(err.is_rate_limited());

        let err2 = ApiError::new(510, "Request too frequent");
        assert!(err2.is_rate_limited());
    }

    /// Test auth error detection.
    #[test]
    fn test_auth_error() {
        let err = ApiError::new(401, "Unauthorized");
        assert!(err.is_auth_error());

        let err2 = ApiError::new(602, "Signature verification failed");
        assert!(err2.is_auth_error());
    }

    /// Test insufficient balance detection.
    #[test]
    fn test_insufficient_balance_error() {
        let err = ApiError::new(10101, "Insufficient balance");
        assert!(err.is_insufficient_balance());

        let err2 = ApiError::new(30004, "Insufficient balance");
        assert!(err2.is_insufficient_balance());
    }

    /// Test error code conversion.
    #[test]
    fn test_error_code_conversion() {
        assert_eq!(
            SpotErrorCode::from_i32(0),
            Some(SpotErrorCode::Success)
        );
        assert_eq!(
            SpotErrorCode::from_i32(429),
            Some(SpotErrorCode::TooManyRequests)
        );
        assert_eq!(
            SpotErrorCode::from_i32(30014),
            Some(SpotErrorCode::InvalidSymbol)
        );
        assert_eq!(SpotErrorCode::from_i32(99999), None);
    }
}

mod rate_limit {
    use mexc_api::rate_limit::{EndpointWeight, RateLimiter};
    use mexc_api::RateLimitConfig;
    use std::time::Duration;

    /// Test rate limiter creation.
    #[test]
    fn test_rate_limiter_creation() {
        let config = RateLimitConfig::default();
        let limiter = RateLimiter::new(config);
        assert_eq!(limiter.config().ip_limit, 500);
    }

    /// Test endpoint weights.
    #[test]
    fn test_endpoint_weights() {
        assert_eq!(EndpointWeight::PING, 1);
        assert_eq!(EndpointWeight::EXCHANGE_INFO, 10);
        assert_eq!(EndpointWeight::TRADES, 5);
        assert_eq!(EndpointWeight::TICKER_24HR_ALL, 40);
        assert_eq!(EndpointWeight::TRADE_FEE, 20);
    }

    /// Test rate limiter acquire.
    #[tokio::test]
    async fn test_rate_limiter_acquire() {
        let config = RateLimitConfig {
            ip_limit: 100,
            ip_window: Duration::from_secs(10),
            burst_allowance: 10,
            min_delay: Duration::from_millis(1),
            ..Default::default()
        };

        let limiter = RateLimiter::new(config);

        // Should be able to acquire immediately
        let weight = limiter.acquire(5).await;
        assert_eq!(weight, 5);

        // Check count
        assert_eq!(limiter.ip_request_count().await, 5);
        assert_eq!(limiter.ip_remaining().await, 95);
    }

    /// Test rate limiter is not rate limited initially.
    #[tokio::test]
    async fn test_rate_limiter_not_rate_limited() {
        let config = RateLimitConfig::default();
        let limiter = RateLimiter::new(config);

        assert!(!limiter.is_rate_limited().await);
    }

    /// Test rate limiter clone shares state.
    #[tokio::test]
    async fn test_rate_limiter_clone() {
        let config = RateLimitConfig {
            ip_limit: 100,
            ip_window: Duration::from_secs(10),
            ..Default::default()
        };

        let limiter1 = RateLimiter::new(config);
        let limiter2 = limiter1.clone();

        limiter1.acquire(10).await;

        // Both should see the same count
        assert_eq!(limiter1.ip_request_count().await, 10);
        assert_eq!(limiter2.ip_request_count().await, 10);
    }
}

mod websocket {
    use mexc_api::websocket::SubscriptionBuilder;
    use mexc_api::types::WsChannel;

    /// Test subscription builder.
    #[test]
    fn test_subscription_builder() {
        let channels = SubscriptionBuilder::new()
            .trades("BTCUSDT")
            .klines("BTCUSDT", "1m")
            .depth("ETHUSDT")
            .depth_limit("BNBUSDT", 20)
            .mini_ticker("XRPUSDT")
            .all_mini_tickers()
            .book_ticker("ADAUSDT")
            .build();

        assert_eq!(channels.len(), 7);

        // Verify specific channels
        assert!(channels.contains(&WsChannel::Trades("BTCUSDT".to_string())));
        assert!(channels.contains(&WsChannel::Klines("BTCUSDT".to_string(), "1m".to_string())));
        assert!(channels.contains(&WsChannel::AllMiniTickers));
    }

    /// Test empty subscription builder.
    #[test]
    fn test_empty_subscription_builder() {
        let channels = SubscriptionBuilder::new().build();
        assert!(channels.is_empty());
    }
}

mod spot {
    use mexc_api::spot::OrderOptions;
    use mexc_api::types::TimeInForce;

    /// Test order options builder.
    #[test]
    fn test_order_options_builder() {
        let options = OrderOptions::new()
            .quantity("0.001")
            .price("50000.00")
            .time_in_force(TimeInForce::Gtc)
            .client_order_id("my_order_123");

        assert!(options.quantity.is_some());
        assert!(options.price.is_some());
        assert_eq!(options.time_in_force, Some(TimeInForce::Gtc));
        assert_eq!(
            options.new_client_order_id,
            Some("my_order_123".to_string())
        );
    }

    /// Test order options with quote quantity.
    #[test]
    fn test_order_options_quote_qty() {
        let options = OrderOptions::new().quote_order_qty("1000.00");

        assert!(options.quote_order_qty.is_some());
        assert!(options.quantity.is_none());
    }

    /// Test order options with stop price.
    #[test]
    fn test_order_options_stop_price() {
        let options = OrderOptions::new()
            .quantity("0.001")
            .stop_price("48000.00")
            .iceberg_qty("0.0001");

        assert!(options.stop_price.is_some());
        assert!(options.iceberg_qty.is_some());
    }
}

// Integration tests that require network access
mod integration {
    use super::*;

    /// Test ping endpoint (requires network).
    #[tokio::test]
    #[ignore = "requires network access"]
    async fn test_ping() {
        let client = MexcClient::public().unwrap();
        let spot = SpotApi::new(client);

        let result = spot.market().ping().await;
        assert!(result.is_ok());
    }

    /// Test server time endpoint (requires network).
    #[tokio::test]
    #[ignore = "requires network access"]
    async fn test_server_time() {
        let client = MexcClient::public().unwrap();
        let spot = SpotApi::new(client);

        let time = spot.market().time().await.unwrap();
        assert!(time.server_time > 0);
    }

    /// Test exchange info endpoint (requires network).
    #[tokio::test]
    #[ignore = "requires network access"]
    async fn test_exchange_info() {
        let client = MexcClient::public().unwrap();
        let spot = SpotApi::new(client);

        let info = spot.market().exchange_info(Some("BTCUSDT"), None).await;
        assert!(info.is_ok());

        let info = info.unwrap();
        assert!(!info.symbols.is_empty());
    }

    /// Test ticker price endpoint (requires network).
    #[tokio::test]
    #[ignore = "requires network access"]
    async fn test_ticker_price() {
        let client = MexcClient::public().unwrap();
        let spot = SpotApi::new(client);

        let ticker = spot.market().ticker_price_single("BTCUSDT").await.unwrap();
        assert_eq!(ticker.symbol, "BTCUSDT");
        assert!(*ticker.price > rust_decimal::Decimal::ZERO);
    }

    /// Test order book endpoint (requires network).
    #[tokio::test]
    #[ignore = "requires network access"]
    async fn test_order_book() {
        let client = MexcClient::public().unwrap();
        let spot = SpotApi::new(client);

        let depth = spot.market().depth("BTCUSDT", Some(5)).await.unwrap();
        assert!(!depth.bids.is_empty());
        assert!(!depth.asks.is_empty());
    }

    /// Test klines endpoint (requires network).
    #[tokio::test]
    #[ignore = "requires network access"]
    async fn test_klines() {
        let client = MexcClient::public().unwrap();
        let spot = SpotApi::new(client);

        let klines = spot
            .market()
            .klines("BTCUSDT", KlineInterval::OneHour, None, None, Some(10))
            .await
            .unwrap();

        assert!(!klines.is_empty());
        assert!(klines.len() <= 10);
    }
}

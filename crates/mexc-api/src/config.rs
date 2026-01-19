//! Configuration for the MEXC API client.

use std::time::Duration;

/// Configuration for the MEXC API client.
#[derive(Debug, Clone)]
pub struct Config {
    /// API key for authentication.
    pub api_key: Option<String>,
    /// API secret for signing requests.
    pub api_secret: Option<String>,
    /// Base URL for REST API (default: https://api.mexc.com).
    pub base_url: String,
    /// WebSocket URL (default: wss://wbs.mexc.com/ws).
    pub ws_url: String,
    /// Request timeout.
    pub timeout: Duration,
    /// Receive window for signed requests (in milliseconds).
    pub recv_window: u64,
    /// Enable rate limiting.
    pub rate_limiting: bool,
    /// Rate limit configuration.
    pub rate_limit_config: RateLimitConfig,
    /// User agent string.
    pub user_agent: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api_key: None,
            api_secret: None,
            base_url: crate::SPOT_BASE_URL.to_string(),
            ws_url: crate::SPOT_WS_URL.to_string(),
            timeout: Duration::from_secs(30),
            recv_window: 5000,
            rate_limiting: true,
            rate_limit_config: RateLimitConfig::default(),
            user_agent: format!("mexc-api-rust/{}", crate::VERSION),
        }
    }
}

impl Config {
    /// Create a new configuration with API credentials.
    pub fn new(api_key: impl Into<String>, api_secret: impl Into<String>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            api_secret: Some(api_secret.into()),
            ..Default::default()
        }
    }

    /// Create a configuration without credentials (public endpoints only).
    pub fn public() -> Self {
        Self::default()
    }

    /// Set the base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set the WebSocket URL.
    pub fn with_ws_url(mut self, url: impl Into<String>) -> Self {
        self.ws_url = url.into();
        self
    }

    /// Set the request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the receive window for signed requests.
    pub fn with_recv_window(mut self, recv_window: u64) -> Self {
        self.recv_window = recv_window;
        self
    }

    /// Enable or disable rate limiting.
    pub fn with_rate_limiting(mut self, enabled: bool) -> Self {
        self.rate_limiting = enabled;
        self
    }

    /// Set the rate limit configuration.
    pub fn with_rate_limit_config(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit_config = config;
        self
    }

    /// Set the user agent.
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = user_agent.into();
        self
    }

    /// Check if credentials are configured.
    pub fn has_credentials(&self) -> bool {
        self.api_key.is_some() && self.api_secret.is_some()
    }
}

/// Rate limit configuration.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum requests per window for IP-based limits.
    pub ip_limit: u32,
    /// Window duration for IP-based limits.
    pub ip_window: Duration,
    /// Maximum requests per window for UID-based limits.
    pub uid_limit: u32,
    /// Window duration for UID-based limits.
    pub uid_window: Duration,
    /// Burst allowance (extra requests allowed in burst).
    pub burst_allowance: u32,
    /// Minimum delay between requests when rate limited.
    pub min_delay: Duration,
    /// Whether to automatically retry on rate limit errors.
    pub auto_retry: bool,
    /// Maximum number of retries.
    pub max_retries: u32,
    /// Backoff multiplier for retries.
    pub backoff_multiplier: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            // MEXC default: 500 requests per 10 seconds
            ip_limit: 500,
            ip_window: Duration::from_secs(10),
            uid_limit: 500,
            uid_window: Duration::from_secs(10),
            burst_allowance: 10,
            min_delay: Duration::from_millis(20),
            auto_retry: true,
            max_retries: 3,
            backoff_multiplier: 2.0,
        }
    }
}

impl RateLimitConfig {
    /// Create a conservative rate limit configuration.
    pub fn conservative() -> Self {
        Self {
            ip_limit: 400,
            ip_window: Duration::from_secs(10),
            uid_limit: 400,
            uid_window: Duration::from_secs(10),
            burst_allowance: 5,
            min_delay: Duration::from_millis(30),
            auto_retry: true,
            max_retries: 5,
            backoff_multiplier: 2.0,
        }
    }

    /// Create an aggressive rate limit configuration (closer to limits).
    pub fn aggressive() -> Self {
        Self {
            ip_limit: 480,
            ip_window: Duration::from_secs(10),
            uid_limit: 480,
            uid_window: Duration::from_secs(10),
            burst_allowance: 15,
            min_delay: Duration::from_millis(10),
            auto_retry: true,
            max_retries: 2,
            backoff_multiplier: 1.5,
        }
    }

    /// Disable rate limiting (not recommended).
    pub fn disabled() -> Self {
        Self {
            ip_limit: u32::MAX,
            ip_window: Duration::from_secs(1),
            uid_limit: u32::MAX,
            uid_window: Duration::from_secs(1),
            burst_allowance: 0,
            min_delay: Duration::ZERO,
            auto_retry: false,
            max_retries: 0,
            backoff_multiplier: 1.0,
        }
    }
}

/// WebSocket configuration.
#[derive(Debug, Clone)]
pub struct WsConfig {
    /// WebSocket URL.
    pub url: String,
    /// Ping interval.
    pub ping_interval: Duration,
    /// Pong timeout.
    pub pong_timeout: Duration,
    /// Reconnect on disconnect.
    pub auto_reconnect: bool,
    /// Maximum reconnect attempts.
    pub max_reconnect_attempts: u32,
    /// Reconnect delay.
    pub reconnect_delay: Duration,
    /// Channel buffer size.
    pub channel_buffer_size: usize,
}

impl Default for WsConfig {
    fn default() -> Self {
        Self {
            url: crate::SPOT_WS_URL.to_string(),
            ping_interval: Duration::from_secs(30),
            pong_timeout: Duration::from_secs(10),
            auto_reconnect: true,
            max_reconnect_attempts: 5,
            reconnect_delay: Duration::from_secs(1),
            channel_buffer_size: 1000,
        }
    }
}

impl WsConfig {
    /// Create a WebSocket config with custom URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Set the ping interval.
    pub fn with_ping_interval(mut self, interval: Duration) -> Self {
        self.ping_interval = interval;
        self
    }

    /// Set auto reconnect behavior.
    pub fn with_auto_reconnect(mut self, enabled: bool) -> Self {
        self.auto_reconnect = enabled;
        self
    }

    /// Set the channel buffer size.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.channel_buffer_size = size;
        self
    }
}

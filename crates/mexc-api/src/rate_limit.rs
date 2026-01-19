//! Rate limiting implementation for the MEXC API client.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

use crate::config::RateLimitConfig;

/// Rate limiter for API requests.
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    /// Request timestamps for IP-based rate limiting.
    ip_requests: Arc<Mutex<VecDeque<Instant>>>,
    /// Request timestamps for UID-based rate limiting.
    uid_requests: Arc<Mutex<VecDeque<Instant>>>,
    /// Whether we're currently in a rate-limited state.
    rate_limited_until: Arc<Mutex<Option<Instant>>>,
}

impl RateLimiter {
    /// Create a new rate limiter with the given configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            ip_requests: Arc::new(Mutex::new(VecDeque::new())),
            uid_requests: Arc::new(Mutex::new(VecDeque::new())),
            rate_limited_until: Arc::new(Mutex::new(None)),
        }
    }

    /// Wait until we can make a request, respecting rate limits.
    ///
    /// Returns the weight that was consumed.
    pub async fn acquire(&self, weight: u32) -> u32 {
        loop {
            // Check if we're in a rate-limited state
            {
                let rate_limited = self.rate_limited_until.lock().await;
                if let Some(until) = *rate_limited {
                    if Instant::now() < until {
                        let wait_time = until - Instant::now();
                        drop(rate_limited);
                        tokio::time::sleep(wait_time).await;
                        continue;
                    }
                }
            }

            // Clean old entries and check current rate
            let now = Instant::now();

            // IP rate limiting
            {
                let mut ip_requests = self.ip_requests.lock().await;

                // Remove entries older than the window
                let cutoff = now - self.config.ip_window;
                while ip_requests.front().is_some_and(|t| *t < cutoff) {
                    ip_requests.pop_front();
                }

                // Check if we need to wait
                let effective_limit = self.config.ip_limit + self.config.burst_allowance;
                if ip_requests.len() as u32 + weight > effective_limit {
                    // Calculate how long to wait
                    if let Some(oldest) = ip_requests.front() {
                        let wait_until = *oldest + self.config.ip_window;
                        if wait_until > now {
                            let wait_time = wait_until - now;
                            drop(ip_requests);
                            tokio::time::sleep(wait_time.max(self.config.min_delay)).await;
                            continue;
                        }
                    }
                }

                // Record this request
                for _ in 0..weight {
                    ip_requests.push_back(now);
                }

                return weight;
            }
        }
    }

    /// Record that we received a rate limit response.
    pub async fn record_rate_limit(&self, retry_after_ms: Option<u64>) {
        let wait_duration = retry_after_ms
            .map(Duration::from_millis)
            .unwrap_or(Duration::from_secs(1));

        let until = Instant::now() + wait_duration;

        let mut rate_limited = self.rate_limited_until.lock().await;
        *rate_limited = Some(until);
    }

    /// Clear the rate limited state.
    pub async fn clear_rate_limit(&self) {
        let mut rate_limited = self.rate_limited_until.lock().await;
        *rate_limited = None;
    }

    /// Get the current request count in the IP window.
    pub async fn ip_request_count(&self) -> usize {
        let now = Instant::now();
        let cutoff = now - self.config.ip_window;

        let mut ip_requests = self.ip_requests.lock().await;
        while ip_requests.front().is_some_and(|t| *t < cutoff) {
            ip_requests.pop_front();
        }

        ip_requests.len()
    }

    /// Get the remaining capacity in the IP window.
    pub async fn ip_remaining(&self) -> u32 {
        let count = self.ip_request_count().await as u32;
        self.config.ip_limit.saturating_sub(count)
    }

    /// Check if rate limiting is currently active.
    pub async fn is_rate_limited(&self) -> bool {
        let rate_limited = self.rate_limited_until.lock().await;
        rate_limited.is_some_and(|until| Instant::now() < until)
    }

    /// Get the configuration.
    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }
}

impl Clone for RateLimiter {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            ip_requests: Arc::clone(&self.ip_requests),
            uid_requests: Arc::clone(&self.uid_requests),
            rate_limited_until: Arc::clone(&self.rate_limited_until),
        }
    }
}

/// Weight values for different endpoint types.
#[derive(Debug, Clone, Copy)]
pub struct EndpointWeight;

impl EndpointWeight {
    /// Ping endpoint weight.
    pub const PING: u32 = 1;
    /// Time endpoint weight.
    pub const TIME: u32 = 1;
    /// Exchange info weight.
    pub const EXCHANGE_INFO: u32 = 10;
    /// Depth endpoint weight.
    pub const DEPTH: u32 = 1;
    /// Trades endpoint weight.
    pub const TRADES: u32 = 5;
    /// Aggregate trades weight.
    pub const AGG_TRADES: u32 = 1;
    /// Klines endpoint weight.
    pub const KLINES: u32 = 1;
    /// Average price weight.
    pub const AVG_PRICE: u32 = 1;
    /// 24hr ticker weight (single).
    pub const TICKER_24HR_SINGLE: u32 = 1;
    /// 24hr ticker weight (all).
    pub const TICKER_24HR_ALL: u32 = 40;
    /// Price ticker weight (single).
    pub const PRICE_SINGLE: u32 = 1;
    /// Price ticker weight (all).
    pub const PRICE_ALL: u32 = 2;
    /// Book ticker weight.
    pub const BOOK_TICKER: u32 = 1;
    /// Account info weight.
    pub const ACCOUNT: u32 = 5;
    /// Order operations weight.
    pub const ORDER: u32 = 1;
    /// My trades weight.
    pub const MY_TRADES: u32 = 5;
    /// Trade fee weight.
    pub const TRADE_FEE: u32 = 20;
    /// Currency config weight.
    pub const CURRENCY_CONFIG: u32 = 10;
    /// Deposit address weight.
    pub const DEPOSIT_ADDRESS: u32 = 10;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter_basic() {
        let config = RateLimitConfig {
            ip_limit: 10,
            ip_window: Duration::from_secs(1),
            burst_allowance: 2,
            ..Default::default()
        };

        let limiter = RateLimiter::new(config);

        // Should be able to acquire immediately
        for _ in 0..10 {
            limiter.acquire(1).await;
        }

        // Check count
        assert_eq!(limiter.ip_request_count().await, 10);
    }

    #[tokio::test]
    async fn test_rate_limiter_remaining() {
        let config = RateLimitConfig {
            ip_limit: 100,
            ip_window: Duration::from_secs(10),
            burst_allowance: 0,
            ..Default::default()
        };

        let limiter = RateLimiter::new(config);

        assert_eq!(limiter.ip_remaining().await, 100);

        limiter.acquire(10).await;

        assert_eq!(limiter.ip_remaining().await, 90);
    }
}

//! HTTP client for MEXC API.

use hmac::{Hmac, Mac};
use reqwest::{Client, Method, RequestBuilder, Response};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::Config;
use crate::error::{ApiError, Error, Result};
use crate::rate_limit::RateLimiter;

type HmacSha256 = Hmac<Sha256>;

/// HTTP client for making requests to the MEXC API.
#[derive(Debug, Clone)]
pub struct MexcClient {
    config: Arc<Config>,
    http: Client,
    rate_limiter: RateLimiter,
}

impl MexcClient {
    /// Create a new client with the given configuration.
    pub fn new(config: Config) -> Result<Self> {
        let http = Client::builder()
            .timeout(config.timeout)
            .user_agent(&config.user_agent)
            .build()?;

        let rate_limiter = RateLimiter::new(config.rate_limit_config.clone());

        Ok(Self {
            config: Arc::new(config),
            http,
            rate_limiter,
        })
    }

    /// Create a client with API credentials.
    pub fn with_credentials(
        api_key: impl Into<String>,
        api_secret: impl Into<String>,
    ) -> Result<Self> {
        Self::new(Config::new(api_key, api_secret))
    }

    /// Create a client for public endpoints only.
    pub fn public() -> Result<Self> {
        Self::new(Config::public())
    }

    /// Get the configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get the rate limiter.
    pub fn rate_limiter(&self) -> &RateLimiter {
        &self.rate_limiter
    }

    /// Get current timestamp in milliseconds.
    fn timestamp_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as i64
    }

    /// Sign a query string using HMAC-SHA256.
    fn sign(&self, query_string: &str) -> Result<String> {
        let secret = self
            .config
            .api_secret
            .as_ref()
            .ok_or_else(|| Error::Authentication("API secret not configured".into()))?;

        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .map_err(|e| Error::Authentication(format!("HMAC error: {e}")))?;

        mac.update(query_string.as_bytes());
        let result = mac.finalize();

        Ok(hex::encode(result.into_bytes()))
    }

    /// Build query string from parameters.
    fn build_query_string(params: &HashMap<String, String>) -> String {
        let mut pairs: Vec<_> = params.iter().collect();
        pairs.sort_by(|a, b| a.0.cmp(b.0));

        pairs
            .into_iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&")
    }

    /// Make a public (unsigned) request.
    pub async fn public_request<T: serde::de::DeserializeOwned>(
        &self,
        method: Method,
        endpoint: &str,
        params: Option<HashMap<String, String>>,
        weight: u32,
    ) -> Result<T> {
        // Apply rate limiting
        if self.config.rate_limiting {
            self.rate_limiter.acquire(weight).await;
        }

        let url = format!("{}/api/v3{}", self.config.base_url, endpoint);

        let mut request = self.http.request(method.clone(), &url);

        // Add API key header if available
        if let Some(ref api_key) = self.config.api_key {
            request = request.header("X-MEXC-APIKEY", api_key);
        }

        // Add query parameters
        if let Some(params) = params {
            request = request.query(&params);
        }

        self.execute_request(request, weight).await
    }

    /// Make a signed (authenticated) request.
    pub async fn signed_request<T: serde::de::DeserializeOwned>(
        &self,
        method: Method,
        endpoint: &str,
        mut params: HashMap<String, String>,
        weight: u32,
    ) -> Result<T> {
        // Ensure we have credentials
        if !self.config.has_credentials() {
            return Err(Error::Authentication("API credentials not configured".into()));
        }

        // Apply rate limiting
        if self.config.rate_limiting {
            self.rate_limiter.acquire(weight).await;
        }

        // Add timestamp
        params.insert("timestamp".to_string(), Self::timestamp_ms().to_string());

        // Add recvWindow if not already present
        if !params.contains_key("recvWindow") {
            params.insert("recvWindow".to_string(), self.config.recv_window.to_string());
        }

        // Build query string and sign
        let query_string = Self::build_query_string(&params);
        let signature = self.sign(&query_string)?;

        // Add signature to params
        params.insert("signature".to_string(), signature);

        let url = format!("{}/api/v3{}", self.config.base_url, endpoint);

        let request = self
            .http
            .request(method, &url)
            .header("X-MEXC-APIKEY", self.config.api_key.as_ref().unwrap())
            .header("Content-Type", "application/json")
            .query(&params);

        self.execute_request(request, weight).await
    }

    /// Execute a request and handle the response.
    async fn execute_request<T: serde::de::DeserializeOwned>(
        &self,
        request: RequestBuilder,
        _weight: u32,
    ) -> Result<T> {
        let mut retries = 0;
        let max_retries = if self.config.rate_limiting {
            self.config.rate_limit_config.max_retries
        } else {
            0
        };

        loop {
            let request_clone = request
                .try_clone()
                .ok_or_else(|| Error::InvalidParameter("Failed to clone request".to_string()))?;

            let response = request_clone.send().await?;

            match self.handle_response(response).await {
                Ok(value) => return Ok(value),
                Err(Error::RateLimited { retry_after_ms }) => {
                    if retries >= max_retries || !self.config.rate_limit_config.auto_retry {
                        return Err(Error::RateLimited { retry_after_ms });
                    }

                    // Record rate limit and wait
                    self.rate_limiter.record_rate_limit(retry_after_ms).await;

                    let wait_ms = retry_after_ms.unwrap_or(1000);
                    let backoff = (self.config.rate_limit_config.backoff_multiplier as u64)
                        .pow(retries as u32);
                    let total_wait = wait_ms * backoff;

                    tracing::warn!(
                        "Rate limited, waiting {}ms before retry {}/{}",
                        total_wait,
                        retries + 1,
                        max_retries
                    );

                    tokio::time::sleep(std::time::Duration::from_millis(total_wait)).await;
                    self.rate_limiter.clear_rate_limit().await;

                    retries += 1;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Handle the API response.
    async fn handle_response<T: serde::de::DeserializeOwned>(
        &self,
        response: Response,
    ) -> Result<T> {
        let status = response.status();

        // Check for rate limit
        if status.as_u16() == 429 {
            let retry_after = response
                .headers()
                .get("Retry-After")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse().ok())
                .map(|s: u64| s * 1000); // Convert seconds to milliseconds

            return Err(Error::RateLimited {
                retry_after_ms: retry_after,
            });
        }

        let body = response.text().await?;

        // Try to parse as error response first
        if !status.is_success() {
            if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&body) {
                return Err(Error::Api(ApiError::new(
                    error_response.code,
                    error_response.msg,
                )));
            }

            return Err(Error::Api(ApiError::new(
                status.as_u16() as i32,
                format!("HTTP {}: {}", status, body),
            )));
        }

        // Try to parse successful response
        // First check if it's an API error wrapped in success status
        if let Ok(error_response) = serde_json::from_str::<ErrorResponse>(&body) {
            if error_response.code != 0 {
                return Err(Error::Api(ApiError::new(
                    error_response.code,
                    error_response.msg,
                )));
            }
        }

        serde_json::from_str(&body).map_err(|e| {
            tracing::error!("Failed to parse response: {}", body);
            Error::Json(e)
        })
    }

    /// Make a GET request to a public endpoint.
    pub async fn get<T: serde::de::DeserializeOwned>(
        &self,
        endpoint: &str,
        params: Option<HashMap<String, String>>,
        weight: u32,
    ) -> Result<T> {
        self.public_request(Method::GET, endpoint, params, weight)
            .await
    }

    /// Make a signed GET request.
    pub async fn signed_get<T: serde::de::DeserializeOwned>(
        &self,
        endpoint: &str,
        params: HashMap<String, String>,
        weight: u32,
    ) -> Result<T> {
        self.signed_request(Method::GET, endpoint, params, weight)
            .await
    }

    /// Make a signed POST request.
    pub async fn signed_post<T: serde::de::DeserializeOwned>(
        &self,
        endpoint: &str,
        params: HashMap<String, String>,
        weight: u32,
    ) -> Result<T> {
        self.signed_request(Method::POST, endpoint, params, weight)
            .await
    }

    /// Make a signed DELETE request.
    pub async fn signed_delete<T: serde::de::DeserializeOwned>(
        &self,
        endpoint: &str,
        params: HashMap<String, String>,
        weight: u32,
    ) -> Result<T> {
        self.signed_request(Method::DELETE, endpoint, params, weight)
            .await
    }
}

/// Error response from the API.
#[derive(Debug, serde::Deserialize)]
struct ErrorResponse {
    code: i32,
    #[serde(alias = "message")]
    msg: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_query_string() {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), "BTCUSDT".to_string());
        params.insert("side".to_string(), "BUY".to_string());
        params.insert("type".to_string(), "LIMIT".to_string());

        let query = MexcClient::build_query_string(&params);

        // Should be sorted alphabetically
        assert!(query.contains("side=BUY"));
        assert!(query.contains("symbol=BTCUSDT"));
        assert!(query.contains("type=LIMIT"));
    }

    #[test]
    fn test_sign() {
        let config = Config::new("test_key", "test_secret");
        let client = MexcClient::new(config).unwrap();

        let signature = client.sign("symbol=BTCUSDT&timestamp=1234567890").unwrap();

        // Signature should be a hex string
        assert_eq!(signature.len(), 64);
        assert!(signature.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_public_client_no_sign() {
        let client = MexcClient::public().unwrap();

        let result = client.sign("test");
        assert!(result.is_err());
    }
}

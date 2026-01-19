//! Error types for the MEXC API client.

use std::fmt;

/// Result type alias for MEXC API operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for the MEXC API client.
#[derive(Debug)]
pub enum Error {
    /// HTTP request failed
    Http(reqwest::Error),
    /// WebSocket error
    WebSocket(tokio_tungstenite::tungstenite::Error),
    /// JSON serialization/deserialization error
    Json(serde_json::Error),
    /// API returned an error response
    Api(ApiError),
    /// Authentication error (missing or invalid credentials)
    Authentication(String),
    /// Rate limit exceeded
    RateLimited {
        /// Retry after this many milliseconds (if provided)
        retry_after_ms: Option<u64>,
    },
    /// Invalid parameter provided
    InvalidParameter(String),
    /// Channel send error (for WebSocket streams)
    ChannelSend(String),
    /// Channel receive error
    ChannelRecv(String),
    /// Connection closed unexpectedly
    ConnectionClosed,
    /// Timeout waiting for response
    Timeout,
    /// URL parsing error
    UrlParse(url::ParseError),
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Http(e) => Some(e),
            Error::WebSocket(e) => Some(e),
            Error::Json(e) => Some(e),
            Error::UrlParse(e) => Some(e),
            _ => None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Http(e) => write!(f, "HTTP error: {e}"),
            Error::WebSocket(e) => write!(f, "WebSocket error: {e}"),
            Error::Json(e) => write!(f, "JSON error: {e}"),
            Error::Api(e) => write!(f, "API error: {e}"),
            Error::Authentication(msg) => write!(f, "Authentication error: {msg}"),
            Error::RateLimited { retry_after_ms } => {
                if let Some(ms) = retry_after_ms {
                    write!(f, "Rate limited, retry after {ms}ms")
                } else {
                    write!(f, "Rate limited")
                }
            }
            Error::InvalidParameter(msg) => write!(f, "Invalid parameter: {msg}"),
            Error::ChannelSend(msg) => write!(f, "Channel send error: {msg}"),
            Error::ChannelRecv(msg) => write!(f, "Channel receive error: {msg}"),
            Error::ConnectionClosed => write!(f, "Connection closed unexpectedly"),
            Error::Timeout => write!(f, "Request timed out"),
            Error::UrlParse(e) => write!(f, "URL parse error: {e}"),
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Self {
        Error::Http(err)
    }
}

impl From<tokio_tungstenite::tungstenite::Error> for Error {
    fn from(err: tokio_tungstenite::tungstenite::Error) -> Self {
        Error::WebSocket(err)
    }
}

impl From<serde_json::Error> for Error {
    fn from(err: serde_json::Error) -> Self {
        Error::Json(err)
    }
}

impl From<url::ParseError> for Error {
    fn from(err: url::ParseError) -> Self {
        Error::UrlParse(err)
    }
}

impl<T> From<tokio::sync::mpsc::error::SendError<T>> for Error {
    fn from(err: tokio::sync::mpsc::error::SendError<T>) -> Self {
        Error::ChannelSend(err.to_string())
    }
}

/// API error returned by MEXC endpoints.
#[derive(Debug, Clone)]
pub struct ApiError {
    /// Error code from the API
    pub code: i32,
    /// Error message
    pub message: String,
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for ApiError {}

impl ApiError {
    /// Create a new API error.
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// Check if this is a rate limit error.
    pub fn is_rate_limited(&self) -> bool {
        self.code == 429 || self.code == 510
    }

    /// Check if this is an authentication error.
    pub fn is_auth_error(&self) -> bool {
        matches!(self.code, 401 | 602 | 700003 | 700006 | 700007)
    }

    /// Check if this is a parameter error.
    pub fn is_param_error(&self) -> bool {
        matches!(self.code, 600 | 601 | 30014 | 30019 | 30020)
    }

    /// Check if this is an insufficient balance error.
    pub fn is_insufficient_balance(&self) -> bool {
        matches!(self.code, 10101 | 30004 | 2005)
    }
}

/// Known MEXC API error codes for spot trading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SpotErrorCode {
    /// Success
    Success = 0,
    /// API key required
    ApiKeyRequired = 400,
    /// No authority
    NoAuthority = 401,
    /// Too many requests
    TooManyRequests = 429,
    /// Signature verification failed
    SignatureVerificationFailed = 602,
    /// Timestamp outside recvWindow
    TimestampOutsideWindow = 700003,
    /// Invalid recvWindow
    InvalidRecvWindow = 700004,
    /// Invalid IP request source
    InvalidIpSource = 700005,
    /// IP not on whitelist
    IpNotWhitelisted = 700006,
    /// No permission for endpoint
    NoPermission = 700007,
    /// User doesn't exist
    UserNotFound = 10001,
    /// Insufficient balance
    InsufficientBalance = 10101,
    /// Pair suspended
    PairSuspended = 30001,
    /// Minimum transaction volume not met
    MinVolumeNotMet = 30002,
    /// Maximum transaction volume exceeded
    MaxVolumeExceeded = 30003,
    /// Insufficient balance for trade
    InsufficientBalanceForTrade = 30004,
    /// Oversold
    Oversold = 30005,
    /// Minimum price violation
    MinPriceViolation = 30010,
    /// Maximum price violation
    MaxPriceViolation = 30011,
    /// Invalid symbol
    InvalidSymbol = 30014,
    /// Order does not exist
    OrderNotFound = 30016,
    /// Order already closed
    OrderAlreadyClosed = 30017,
    /// Price precision exceeded
    PricePrecisionExceeded = 30019,
    /// Quantity precision exceeded
    QuantityPrecisionExceeded = 30020,
    /// Minimum quantity not met
    MinQuantityNotMet = 30021,
    /// Maximum quantity exceeded
    MaxQuantityExceeded = 30022,
    /// Maximum open orders exceeded
    MaxOpenOrdersExceeded = 30024,
    /// Maximum order limit exceeded
    MaxOrderLimitExceeded = 30029,
}

impl SpotErrorCode {
    /// Get the error code as an i32.
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    /// Try to convert an i32 to a SpotErrorCode.
    pub fn from_i32(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Success),
            400 => Some(Self::ApiKeyRequired),
            401 => Some(Self::NoAuthority),
            429 => Some(Self::TooManyRequests),
            602 => Some(Self::SignatureVerificationFailed),
            700003 => Some(Self::TimestampOutsideWindow),
            700004 => Some(Self::InvalidRecvWindow),
            700005 => Some(Self::InvalidIpSource),
            700006 => Some(Self::IpNotWhitelisted),
            700007 => Some(Self::NoPermission),
            10001 => Some(Self::UserNotFound),
            10101 => Some(Self::InsufficientBalance),
            30001 => Some(Self::PairSuspended),
            30002 => Some(Self::MinVolumeNotMet),
            30003 => Some(Self::MaxVolumeExceeded),
            30004 => Some(Self::InsufficientBalanceForTrade),
            30005 => Some(Self::Oversold),
            30010 => Some(Self::MinPriceViolation),
            30011 => Some(Self::MaxPriceViolation),
            30014 => Some(Self::InvalidSymbol),
            30016 => Some(Self::OrderNotFound),
            30017 => Some(Self::OrderAlreadyClosed),
            30019 => Some(Self::PricePrecisionExceeded),
            30020 => Some(Self::QuantityPrecisionExceeded),
            30021 => Some(Self::MinQuantityNotMet),
            30022 => Some(Self::MaxQuantityExceeded),
            30024 => Some(Self::MaxOpenOrdersExceeded),
            30029 => Some(Self::MaxOrderLimitExceeded),
            _ => None,
        }
    }
}

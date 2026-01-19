//! Common types used across the API.

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Server time response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ServerTime {
    /// Server timestamp in milliseconds.
    pub server_time: i64,
}

/// Empty response (e.g., from ping).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Empty {}

/// Order side (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum OrderSide {
    /// Buy order
    Buy,
    /// Sell order
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

/// Order type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OrderType {
    /// Limit order
    Limit,
    /// Market order
    Market,
    /// Limit maker (post-only)
    LimitMaker,
    /// Immediate or cancel
    #[serde(rename = "IMMEDIATE_OR_CANCEL")]
    ImmediateOrCancel,
    /// Fill or kill
    #[serde(rename = "FILL_OR_KILL")]
    FillOrKill,
}

impl std::fmt::Display for OrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderType::Limit => write!(f, "LIMIT"),
            OrderType::Market => write!(f, "MARKET"),
            OrderType::LimitMaker => write!(f, "LIMIT_MAKER"),
            OrderType::ImmediateOrCancel => write!(f, "IMMEDIATE_OR_CANCEL"),
            OrderType::FillOrKill => write!(f, "FILL_OR_KILL"),
        }
    }
}

/// Order status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OrderStatus {
    /// Order accepted
    New,
    /// Partially executed
    PartiallyFilled,
    /// Fully executed
    Filled,
    /// Canceled by user
    Canceled,
    /// Pending cancellation
    PendingCancel,
    /// Order rejected
    Rejected,
    /// Order expired
    Expired,
}

/// Time in force for orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum TimeInForce {
    /// Good till canceled
    #[serde(rename = "GTC")]
    Gtc,
    /// Immediate or cancel
    #[serde(rename = "IOC")]
    Ioc,
    /// Fill or kill
    #[serde(rename = "FOK")]
    Fok,
}

impl std::fmt::Display for TimeInForce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeInForce::Gtc => write!(f, "GTC"),
            TimeInForce::Ioc => write!(f, "IOC"),
            TimeInForce::Fok => write!(f, "FOK"),
        }
    }
}

/// Kline/candlestick interval.
/// Note: MEXC uses different interval strings than other exchanges.
/// For example, hourly is "60m" not "1h".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum KlineInterval {
    /// 1 minute
    #[serde(rename = "1m")]
    OneMinute,
    /// 5 minutes
    #[serde(rename = "5m")]
    FiveMinutes,
    /// 15 minutes
    #[serde(rename = "15m")]
    FifteenMinutes,
    /// 30 minutes
    #[serde(rename = "30m")]
    ThirtyMinutes,
    /// 1 hour (60m for MEXC)
    #[serde(rename = "60m")]
    OneHour,
    /// 4 hours (4h for MEXC)
    #[serde(rename = "4h")]
    FourHours,
    /// 8 hours (8h for MEXC)
    #[serde(rename = "8h")]
    EightHours,
    /// 1 day
    #[serde(rename = "1d")]
    OneDay,
    /// 1 week
    #[serde(rename = "1W")]
    OneWeek,
    /// 1 month
    #[serde(rename = "1M")]
    OneMonth,
}

impl std::fmt::Display for KlineInterval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KlineInterval::OneMinute => write!(f, "1m"),
            KlineInterval::FiveMinutes => write!(f, "5m"),
            KlineInterval::FifteenMinutes => write!(f, "15m"),
            KlineInterval::ThirtyMinutes => write!(f, "30m"),
            KlineInterval::OneHour => write!(f, "60m"),
            KlineInterval::FourHours => write!(f, "4h"),
            KlineInterval::EightHours => write!(f, "8h"),
            KlineInterval::OneDay => write!(f, "1d"),
            KlineInterval::OneWeek => write!(f, "1W"),
            KlineInterval::OneMonth => write!(f, "1M"),
        }
    }
}

/// Response type for new orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum NewOrderRespType {
    /// Acknowledgment only
    Ack,
    /// Result with order info
    Result,
    /// Full response with fills
    Full,
}

impl std::fmt::Display for NewOrderRespType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NewOrderRespType::Ack => write!(f, "ACK"),
            NewOrderRespType::Result => write!(f, "RESULT"),
            NewOrderRespType::Full => write!(f, "FULL"),
        }
    }
}

/// Decimal string wrapper that deserializes from string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StringDecimal(pub Decimal);

impl StringDecimal {
    /// Create a new StringDecimal.
    pub fn new(value: Decimal) -> Self {
        Self(value)
    }

    /// Create a StringDecimal from a string.
    pub fn from_string(s: String) -> Self {
        Self(s.parse().unwrap_or_default())
    }

    /// Get the inner Decimal value.
    pub fn inner(&self) -> Decimal {
        self.0
    }
}

impl From<Decimal> for StringDecimal {
    fn from(value: Decimal) -> Self {
        Self(value)
    }
}

impl From<StringDecimal> for Decimal {
    fn from(value: StringDecimal) -> Self {
        value.0
    }
}

impl std::ops::Deref for StringDecimal {
    type Target = Decimal;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'de> Deserialize<'de> for StringDecimal {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        // Try to deserialize as string first, then as number
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum StringOrNumber {
            String(String),
            Number(f64),
        }

        match StringOrNumber::deserialize(deserializer)? {
            StringOrNumber::String(s) => {
                s.parse::<Decimal>()
                    .map(StringDecimal)
                    .map_err(|e| D::Error::custom(format!("invalid decimal: {e}")))
            }
            StringOrNumber::Number(n) => {
                Decimal::try_from(n)
                    .map(StringDecimal)
                    .map_err(|e| D::Error::custom(format!("invalid decimal: {e}")))
            }
        }
    }
}

impl Serialize for StringDecimal {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0.to_string())
    }
}

impl std::fmt::Display for StringDecimal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

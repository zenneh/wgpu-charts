//! Protocol Buffers message definitions for MEXC WebSocket API.
//!
//! These are manually defined based on MEXC's proto files at:
//! https://github.com/mexcdevelop/websocket-proto

#![allow(missing_docs)]

use prost::Message;

/// Wrapper message for all MEXC WebSocket push data.
#[derive(Clone, PartialEq, Message)]
pub struct PushDataV3ApiWrapper {
    /// Channel identifier (e.g., "spot@public.kline.v3.api.pb@BTCUSDT@Min1")
    #[prost(string, tag = "1")]
    pub channel: String,

    /// Trading pair symbol (e.g., "BTCUSDT")
    #[prost(string, optional, tag = "3")]
    pub symbol: Option<String>,

    /// Trading pair identifier
    #[prost(string, optional, tag = "4")]
    pub symbol_id: Option<String>,

    /// Message creation timestamp (milliseconds)
    #[prost(int64, optional, tag = "5")]
    pub create_time: Option<i64>,

    /// Message send timestamp (milliseconds)
    #[prost(int64, optional, tag = "6")]
    pub send_time: Option<i64>,

    /// Body containing the actual data (oneof)
    #[prost(oneof = "push_data_body::Body", tags = "301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315")]
    pub body: Option<push_data_body::Body>,
}

/// Body variants for PushDataV3ApiWrapper.
pub mod push_data_body {
    #[derive(Clone, PartialEq, ::prost::Oneof)]
    pub enum Body {
        /// Public deals data
        #[prost(message, tag = "301")]
        PublicDeals(super::PublicDealsV3Api),

        /// Public increase depths data
        #[prost(message, tag = "302")]
        PublicIncreaseDepths(super::PublicIncreaseDepthsV3Api),

        /// Public limit depths data
        #[prost(message, tag = "303")]
        PublicLimitDepths(super::PublicLimitDepthsV3Api),

        /// Private orders data
        #[prost(message, tag = "304")]
        PrivateOrders(super::PrivateOrdersV3Api),

        /// Public book ticker data
        #[prost(message, tag = "305")]
        PublicBookTicker(super::PublicBookTickerV3Api),

        /// Private deals data
        #[prost(message, tag = "306")]
        PrivateDeals(super::PrivateDealsV3Api),

        /// Private account data
        #[prost(message, tag = "307")]
        PrivateAccount(super::PrivateAccountV3Api),

        /// Public spot kline data
        #[prost(message, tag = "308")]
        PublicSpotKline(super::PublicSpotKlineV3Api),

        /// Public mini ticker data
        #[prost(message, tag = "309")]
        PublicMiniTicker(super::PublicMiniTickerV3Api),

        /// Public mini tickers data (batch)
        #[prost(message, tag = "310")]
        PublicMiniTickers(super::PublicMiniTickersV3Api),

        /// Public book ticker batch data
        #[prost(message, tag = "311")]
        PublicBookTickerBatch(super::PublicBookTickerBatchV3Api),

        /// Public increase depths batch data
        #[prost(message, tag = "312")]
        PublicIncreaseDepthsBatch(super::PublicIncreaseDepthsBatchV3Api),

        /// Public aggregate depths data
        #[prost(message, tag = "313")]
        PublicAggreDepths(super::PublicAggreDepthsV3Api),

        /// Public aggregate deals data
        #[prost(message, tag = "314")]
        PublicAggreDeals(super::PublicAggreDealsV3Api),

        /// Public aggregate book ticker data
        #[prost(message, tag = "315")]
        PublicAggreBookTicker(super::PublicAggreBookTickerV3Api),
    }
}

/// Spot kline (candlestick) data.
#[derive(Clone, PartialEq, Message)]
pub struct PublicSpotKlineV3Api {
    /// K-line interval (Min1, Min5, Min15, Min30, Min60, Hour4, Hour8, Day1, Week1, Month1)
    #[prost(string, tag = "1")]
    pub interval: String,

    /// Window start timestamp (seconds)
    #[prost(int64, tag = "2")]
    pub window_start: i64,

    /// Opening price
    #[prost(string, tag = "3")]
    pub opening_price: String,

    /// Closing price
    #[prost(string, tag = "4")]
    pub closing_price: String,

    /// Highest price
    #[prost(string, tag = "5")]
    pub highest_price: String,

    /// Lowest price
    #[prost(string, tag = "6")]
    pub lowest_price: String,

    /// Trading volume
    #[prost(string, tag = "7")]
    pub volume: String,

    /// Trading amount
    #[prost(string, tag = "8")]
    pub amount: String,

    /// Window end timestamp (seconds)
    #[prost(int64, tag = "9")]
    pub window_end: i64,
}

// Placeholder types for other message types (not fully implemented)
// These are needed for the oneof to compile

/// Public deals (trades) data.
#[derive(Clone, PartialEq, Message)]
pub struct PublicDealsV3Api {
    #[prost(message, repeated, tag = "1")]
    pub deals: Vec<PublicDealV3>,
    #[prost(string, tag = "2")]
    pub event_type: String,
}

/// A single trade/deal entry.
#[derive(Clone, PartialEq, Message)]
pub struct PublicDealV3 {
    #[prost(string, tag = "1")]
    pub price: String,
    #[prost(string, tag = "2")]
    pub quantity: String,
    /// 1 = buy, 2 = sell
    #[prost(int32, tag = "3")]
    pub trade_type: i32,
    #[prost(int64, tag = "4")]
    pub time: i64,
}

#[derive(Clone, PartialEq, Message)]
pub struct PublicIncreaseDepthsV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

/// Public limit depth snapshot data.
#[derive(Clone, PartialEq, Message)]
pub struct PublicLimitDepthsV3Api {
    #[prost(message, repeated, tag = "1")]
    pub bids: Vec<DepthEntryV3>,
    #[prost(message, repeated, tag = "2")]
    pub asks: Vec<DepthEntryV3>,
    #[prost(string, tag = "3")]
    pub version: String,
    #[prost(string, tag = "4")]
    pub event_type: String,
}

/// A single depth level entry (price + quantity).
#[derive(Clone, PartialEq, Message)]
pub struct DepthEntryV3 {
    #[prost(string, tag = "1")]
    pub price: String,
    #[prost(string, tag = "2")]
    pub quantity: String,
}

#[derive(Clone, PartialEq, Message)]
pub struct PrivateOrdersV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct PublicBookTickerV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct PrivateDealsV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct PrivateAccountV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct PublicMiniTickerV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct PublicMiniTickersV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct PublicBookTickerBatchV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct PublicIncreaseDepthsBatchV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct PublicAggreDepthsV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct PublicAggreDealsV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

#[derive(Clone, PartialEq, Message)]
pub struct PublicAggreBookTickerV3Api {
    #[prost(bytes = "vec", tag = "1")]
    pub data: Vec<u8>,
}

/// Decode a protobuf message from binary data.
pub fn decode_push_data(data: &[u8]) -> Result<PushDataV3ApiWrapper, prost::DecodeError> {
    PushDataV3ApiWrapper::decode(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kline_struct() {
        let kline = PublicSpotKlineV3Api {
            interval: "Min1".to_string(),
            window_start: 1700000000,
            opening_price: "100.5".to_string(),
            closing_price: "101.0".to_string(),
            highest_price: "102.0".to_string(),
            lowest_price: "99.5".to_string(),
            volume: "1000.0".to_string(),
            amount: "100500.0".to_string(),
            window_end: 1700000060,
        };
        assert_eq!(kline.interval, "Min1");
    }
}

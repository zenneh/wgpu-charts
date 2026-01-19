//! Market data types.

use super::common::{OrderType, StringDecimal};
use serde::{Deserialize, Serialize};

/// Exchange information response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExchangeInfo {
    /// Server timezone.
    pub timezone: String,
    /// Server time.
    pub server_time: i64,
    /// Rate limit rules.
    #[serde(default)]
    pub rate_limits: Vec<RateLimit>,
    /// Trading symbols.
    pub symbols: Vec<SymbolInfo>,
}

/// Rate limit information.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RateLimit {
    /// Rate limit type (REQUEST_WEIGHT, ORDERS, etc.).
    pub rate_limit_type: String,
    /// Interval (SECOND, MINUTE, DAY).
    pub interval: String,
    /// Interval number.
    pub interval_num: i32,
    /// Request limit.
    pub limit: i32,
}

/// Symbol trading information.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SymbolInfo {
    /// Symbol name (e.g., "BTCUSDT").
    pub symbol: String,
    /// Trading status.
    pub status: String,
    /// Base asset (e.g., "BTC").
    pub base_asset: String,
    /// Base asset precision.
    pub base_asset_precision: i32,
    /// Quote asset (e.g., "USDT").
    pub quote_asset: String,
    /// Quote precision.
    pub quote_precision: i32,
    /// Quote asset precision.
    pub quote_asset_precision: i32,
    /// Base size precision.
    #[serde(default)]
    pub base_size_precision: Option<StringDecimal>,
    /// Allowed order types.
    pub order_types: Vec<OrderType>,
    /// Is spot trading allowed.
    #[serde(default)]
    pub is_spot_trading_allowed: bool,
    /// Is margin trading allowed.
    #[serde(default)]
    pub is_margin_trading_allowed: bool,
    /// Quote amount precision.
    #[serde(default)]
    pub quote_amount_precision: Option<StringDecimal>,
    /// Base commission precision.
    #[serde(default)]
    pub base_commission_precision: Option<i32>,
    /// Quote commission precision.
    #[serde(default)]
    pub quote_commission_precision: Option<i32>,
    /// Trading permissions.
    #[serde(default)]
    pub permissions: Vec<String>,
    /// Max quote amount.
    #[serde(default)]
    pub max_quote_amount: Option<StringDecimal>,
    /// Maker commission rate.
    #[serde(default)]
    pub maker_commission: Option<StringDecimal>,
    /// Taker commission rate.
    #[serde(default)]
    pub taker_commission: Option<StringDecimal>,
    /// Full name of the symbol.
    #[serde(default)]
    pub full_name: Option<String>,
}

/// Order book depth response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderBook {
    /// Last update ID.
    pub last_update_id: i64,
    /// Bid orders [price, quantity].
    pub bids: Vec<(StringDecimal, StringDecimal)>,
    /// Ask orders [price, quantity].
    pub asks: Vec<(StringDecimal, StringDecimal)>,
}

/// Single trade.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Trade {
    /// Trade ID.
    pub id: i64,
    /// Price.
    pub price: StringDecimal,
    /// Quantity.
    pub qty: StringDecimal,
    /// Quote quantity.
    pub quote_qty: StringDecimal,
    /// Trade time.
    pub time: i64,
    /// Was the buyer the maker?
    pub is_buyer_maker: bool,
    /// Is best price match?
    pub is_best_match: bool,
}

/// Aggregated trade.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AggTrade {
    /// Aggregate trade ID.
    #[serde(rename = "a")]
    pub agg_id: i64,
    /// Price.
    #[serde(rename = "p")]
    pub price: StringDecimal,
    /// Quantity.
    #[serde(rename = "q")]
    pub qty: StringDecimal,
    /// First trade ID.
    #[serde(rename = "f")]
    pub first_id: i64,
    /// Last trade ID.
    #[serde(rename = "l")]
    pub last_id: i64,
    /// Timestamp.
    #[serde(rename = "T")]
    pub time: i64,
    /// Was the buyer the maker?
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
    /// Is best price match?
    #[serde(rename = "M")]
    pub is_best_match: bool,
}

/// Kline/candlestick data.
#[derive(Debug, Clone, Serialize)]
pub struct Kline {
    /// Open time.
    pub open_time: i64,
    /// Open price.
    pub open: StringDecimal,
    /// High price.
    pub high: StringDecimal,
    /// Low price.
    pub low: StringDecimal,
    /// Close price.
    pub close: StringDecimal,
    /// Volume.
    pub volume: StringDecimal,
    /// Close time.
    pub close_time: i64,
    /// Quote asset volume.
    pub quote_volume: StringDecimal,
    /// Number of trades.
    pub trades: i64,
    /// Taker buy base volume.
    pub taker_buy_base_volume: StringDecimal,
    /// Taker buy quote volume.
    pub taker_buy_quote_volume: StringDecimal,
}

// Custom deserializer for Kline since it comes as an array
impl<'de> Deserialize<'de> for Kline {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        let arr: Vec<serde_json::Value> = Vec::deserialize(deserializer)?;

        if arr.len() < 11 {
            return Err(D::Error::custom("kline array too short"));
        }

        let parse_decimal = |v: &serde_json::Value| -> Result<StringDecimal, D::Error> {
            match v {
                serde_json::Value::String(s) => s
                    .parse::<rust_decimal::Decimal>()
                    .map(StringDecimal)
                    .map_err(|e| D::Error::custom(format!("invalid decimal: {e}"))),
                serde_json::Value::Number(n) => n
                    .as_f64()
                    .ok_or_else(|| D::Error::custom("invalid number"))
                    .and_then(|f| {
                        rust_decimal::Decimal::try_from(f)
                            .map(StringDecimal)
                            .map_err(|e| D::Error::custom(format!("invalid decimal: {e}")))
                    }),
                _ => Err(D::Error::custom("expected string or number")),
            }
        };

        let parse_i64 = |v: &serde_json::Value| -> Result<i64, D::Error> {
            v.as_i64()
                .ok_or_else(|| D::Error::custom("expected integer"))
        };

        Ok(Kline {
            open_time: parse_i64(&arr[0])?,
            open: parse_decimal(&arr[1])?,
            high: parse_decimal(&arr[2])?,
            low: parse_decimal(&arr[3])?,
            close: parse_decimal(&arr[4])?,
            volume: parse_decimal(&arr[5])?,
            close_time: parse_i64(&arr[6])?,
            quote_volume: parse_decimal(&arr[7])?,
            trades: parse_i64(&arr[8])?,
            taker_buy_base_volume: parse_decimal(&arr[9])?,
            taker_buy_quote_volume: parse_decimal(&arr[10])?,
        })
    }
}

/// Average price response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AvgPrice {
    /// Minutes the average was calculated over.
    pub mins: i32,
    /// Average price.
    pub price: StringDecimal,
}

/// 24hr ticker statistics.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Ticker24hr {
    /// Symbol.
    pub symbol: String,
    /// Price change.
    pub price_change: StringDecimal,
    /// Price change percent.
    pub price_change_percent: StringDecimal,
    /// Previous close price.
    pub prev_close_price: StringDecimal,
    /// Last price.
    pub last_price: StringDecimal,
    /// Best bid price.
    pub bid_price: StringDecimal,
    /// Best bid quantity.
    pub bid_qty: StringDecimal,
    /// Best ask price.
    pub ask_price: StringDecimal,
    /// Best ask quantity.
    pub ask_qty: StringDecimal,
    /// Open price.
    pub open_price: StringDecimal,
    /// High price.
    pub high_price: StringDecimal,
    /// Low price.
    pub low_price: StringDecimal,
    /// Base volume.
    pub volume: StringDecimal,
    /// Quote volume.
    pub quote_volume: StringDecimal,
    /// Open time.
    pub open_time: i64,
    /// Close time.
    pub close_time: i64,
    /// Number of trades.
    #[serde(default)]
    pub count: Option<i64>,
}

/// Price ticker.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PriceTicker {
    /// Symbol.
    pub symbol: String,
    /// Price.
    pub price: StringDecimal,
}

/// Book ticker (best bid/ask).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BookTicker {
    /// Symbol.
    pub symbol: String,
    /// Best bid price.
    pub bid_price: StringDecimal,
    /// Best bid quantity.
    pub bid_qty: StringDecimal,
    /// Best ask price.
    pub ask_price: StringDecimal,
    /// Best ask quantity.
    pub ask_qty: StringDecimal,
}

/// Default symbols response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DefaultSymbols {
    /// List of default symbol names.
    pub data: Vec<String>,
}

//! WebSocket message types.

use super::common::{OrderSide, OrderStatus, OrderType, StringDecimal};
use serde::{Deserialize, Serialize};

/// WebSocket subscription method.
#[derive(Debug, Clone, Serialize)]
pub struct WsSubscription {
    /// Method (SUBSCRIPTION or UNSUBSCRIPTION).
    pub method: String,
    /// Parameters (channel names).
    pub params: Vec<String>,
}

impl WsSubscription {
    /// Create a new subscription.
    pub fn subscribe(channels: Vec<String>) -> Self {
        Self {
            method: "SUBSCRIPTION".to_string(),
            params: channels,
        }
    }

    /// Create an unsubscription.
    pub fn unsubscribe(channels: Vec<String>) -> Self {
        Self {
            method: "UNSUBSCRIPTION".to_string(),
            params: channels,
        }
    }
}

/// Generic WebSocket message wrapper.
#[derive(Debug, Clone, Deserialize)]
pub struct WsMessage<T> {
    /// Channel name.
    #[serde(rename = "c")]
    pub channel: String,
    /// Data payload.
    #[serde(rename = "d")]
    pub data: T,
    /// Symbol.
    #[serde(rename = "s")]
    pub symbol: Option<String>,
    /// Timestamp.
    #[serde(rename = "t")]
    pub timestamp: i64,
}

/// WebSocket trade data.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WsTrade {
    /// Price.
    #[serde(rename = "p")]
    pub price: StringDecimal,
    /// Quantity.
    #[serde(rename = "v")]
    pub quantity: StringDecimal,
    /// Side (1 = buy, 2 = sell).
    #[serde(rename = "S")]
    pub side: i32,
    /// Timestamp.
    #[serde(rename = "t")]
    pub time: i64,
}

impl WsTrade {
    /// Check if this is a buy trade.
    pub fn is_buy(&self) -> bool {
        self.side == 1
    }

    /// Check if this is a sell trade.
    pub fn is_sell(&self) -> bool {
        self.side == 2
    }
}

/// WebSocket trades data (multiple trades).
#[derive(Debug, Clone, Deserialize)]
pub struct WsTradesData {
    /// List of trades.
    pub deals: Vec<WsTrade>,
    /// Event type.
    #[serde(rename = "e")]
    pub event: String,
}

/// WebSocket kline data.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WsKline {
    /// Open time.
    #[serde(rename = "t")]
    pub time: i64,
    /// Open price.
    #[serde(rename = "o")]
    pub open: StringDecimal,
    /// High price.
    #[serde(rename = "h")]
    pub high: StringDecimal,
    /// Low price.
    #[serde(rename = "l")]
    pub low: StringDecimal,
    /// Close price.
    #[serde(rename = "c")]
    pub close: StringDecimal,
    /// Volume.
    #[serde(rename = "v")]
    pub volume: StringDecimal,
    /// Quote volume.
    #[serde(rename = "a")]
    pub quote_volume: StringDecimal,
}

/// WebSocket kline data wrapper.
#[derive(Debug, Clone, Deserialize)]
pub struct WsKlineData {
    /// Kline data.
    #[serde(rename = "k")]
    pub kline: WsKline,
    /// Event type.
    #[serde(rename = "e")]
    pub event: String,
}

/// WebSocket depth update.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WsDepthUpdate {
    /// Bids (price, quantity).
    pub bids: Vec<WsDepthLevel>,
    /// Asks (price, quantity).
    pub asks: Vec<WsDepthLevel>,
}

/// Single depth level.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WsDepthLevel {
    /// Price.
    #[serde(rename = "p")]
    pub price: StringDecimal,
    /// Quantity.
    #[serde(rename = "v")]
    pub quantity: StringDecimal,
}

/// WebSocket depth data wrapper.
#[derive(Debug, Clone, Deserialize)]
pub struct WsDepthData {
    /// Bids.
    pub bids: Vec<WsDepthLevel>,
    /// Asks.
    pub asks: Vec<WsDepthLevel>,
    /// Event type.
    #[serde(rename = "e")]
    pub event: String,
    /// First update ID.
    #[serde(rename = "r")]
    pub version: Option<String>,
}

/// WebSocket mini ticker.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WsMiniTicker {
    /// Symbol.
    #[serde(rename = "s")]
    pub symbol: String,
    /// Close price.
    #[serde(rename = "c")]
    pub close: StringDecimal,
    /// Open price.
    #[serde(rename = "o")]
    pub open: StringDecimal,
    /// High price.
    #[serde(rename = "h")]
    pub high: StringDecimal,
    /// Low price.
    #[serde(rename = "l")]
    pub low: StringDecimal,
    /// Base volume.
    #[serde(rename = "v")]
    pub volume: StringDecimal,
    /// Quote volume.
    #[serde(rename = "q")]
    pub quote_volume: StringDecimal,
}

/// WebSocket book ticker.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WsBookTicker {
    /// Symbol.
    #[serde(rename = "s")]
    pub symbol: String,
    /// Best bid price.
    #[serde(rename = "b")]
    pub bid_price: StringDecimal,
    /// Best bid quantity.
    #[serde(rename = "B")]
    pub bid_qty: StringDecimal,
    /// Best ask price.
    #[serde(rename = "a")]
    pub ask_price: StringDecimal,
    /// Best ask quantity.
    #[serde(rename = "A")]
    pub ask_qty: StringDecimal,
}

/// WebSocket account update.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WsAccountUpdate {
    /// Event type.
    #[serde(rename = "e")]
    pub event: String,
    /// Event time.
    #[serde(rename = "E")]
    pub event_time: i64,
    /// Balances.
    #[serde(rename = "B", default)]
    pub balances: Vec<WsBalance>,
}

/// WebSocket balance.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WsBalance {
    /// Asset.
    #[serde(rename = "a")]
    pub asset: String,
    /// Free balance.
    #[serde(rename = "f")]
    pub free: StringDecimal,
    /// Locked balance.
    #[serde(rename = "l")]
    pub locked: StringDecimal,
}

/// WebSocket order update.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WsOrderUpdate {
    /// Event type.
    #[serde(rename = "e")]
    pub event: String,
    /// Event time.
    #[serde(rename = "E")]
    pub event_time: i64,
    /// Symbol.
    #[serde(rename = "s")]
    pub symbol: String,
    /// Client order ID.
    #[serde(rename = "c")]
    pub client_order_id: Option<String>,
    /// Side.
    #[serde(rename = "S")]
    pub side: OrderSide,
    /// Order type.
    #[serde(rename = "o")]
    pub order_type: OrderType,
    /// Time in force.
    #[serde(rename = "f")]
    pub time_in_force: Option<String>,
    /// Original quantity.
    #[serde(rename = "q")]
    pub original_quantity: StringDecimal,
    /// Original price.
    #[serde(rename = "p")]
    pub price: StringDecimal,
    /// Execution type.
    #[serde(rename = "x")]
    pub execution_type: String,
    /// Order status.
    #[serde(rename = "X")]
    pub status: OrderStatus,
    /// Order ID.
    #[serde(rename = "i")]
    pub order_id: String,
    /// Last executed quantity.
    #[serde(rename = "l")]
    pub last_executed_qty: StringDecimal,
    /// Cumulative filled quantity.
    #[serde(rename = "z")]
    pub cumulative_filled_qty: StringDecimal,
    /// Last executed price.
    #[serde(rename = "L")]
    pub last_executed_price: StringDecimal,
    /// Commission amount.
    #[serde(rename = "n")]
    pub commission: Option<StringDecimal>,
    /// Commission asset.
    #[serde(rename = "N")]
    pub commission_asset: Option<String>,
    /// Transaction time.
    #[serde(rename = "T")]
    pub transaction_time: i64,
    /// Trade ID.
    #[serde(rename = "t")]
    pub trade_id: Option<i64>,
    /// Order creation time.
    #[serde(rename = "O")]
    pub order_creation_time: i64,
    /// Quote order quantity.
    #[serde(rename = "Q")]
    pub quote_order_qty: Option<StringDecimal>,
    /// Working time.
    #[serde(rename = "W")]
    pub working_time: Option<i64>,
}

/// WebSocket trade execution.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WsTradeUpdate {
    /// Event type.
    #[serde(rename = "e")]
    pub event: String,
    /// Event time.
    #[serde(rename = "E")]
    pub event_time: i64,
    /// Symbol.
    #[serde(rename = "s")]
    pub symbol: String,
    /// Trade ID.
    #[serde(rename = "t")]
    pub trade_id: i64,
    /// Order ID.
    #[serde(rename = "i")]
    pub order_id: String,
    /// Price.
    #[serde(rename = "p")]
    pub price: StringDecimal,
    /// Quantity.
    #[serde(rename = "q")]
    pub quantity: StringDecimal,
    /// Commission.
    #[serde(rename = "n")]
    pub commission: StringDecimal,
    /// Commission asset.
    #[serde(rename = "N")]
    pub commission_asset: String,
    /// Transaction time.
    #[serde(rename = "T")]
    pub time: i64,
    /// Side.
    #[serde(rename = "S")]
    pub side: OrderSide,
    /// Is buyer.
    #[serde(rename = "m")]
    pub is_buyer: bool,
}

/// Parsed WebSocket event.
#[derive(Debug, Clone)]
pub enum WsEvent {
    /// Trade update.
    Trade {
        /// Symbol.
        symbol: String,
        /// Trades.
        trades: Vec<WsTrade>,
        /// Timestamp.
        timestamp: i64,
    },
    /// Kline update.
    Kline {
        /// Symbol.
        symbol: String,
        /// Interval.
        interval: String,
        /// Kline data.
        kline: WsKline,
        /// Timestamp.
        timestamp: i64,
    },
    /// Depth update.
    Depth {
        /// Symbol.
        symbol: String,
        /// Bids.
        bids: Vec<WsDepthLevel>,
        /// Asks.
        asks: Vec<WsDepthLevel>,
        /// Version.
        version: Option<String>,
        /// Timestamp.
        timestamp: i64,
    },
    /// Mini ticker update.
    MiniTicker {
        /// Ticker data.
        ticker: WsMiniTicker,
        /// Timestamp.
        timestamp: i64,
    },
    /// Book ticker update.
    BookTicker {
        /// Ticker data.
        ticker: WsBookTicker,
        /// Timestamp.
        timestamp: i64,
    },
    /// Account update (private).
    Account(WsAccountUpdate),
    /// Order update (private).
    Order(WsOrderUpdate),
    /// Trade execution (private).
    TradeExecution(WsTradeUpdate),
    /// Unknown/unparsed event.
    Unknown {
        /// Raw JSON.
        raw: String,
    },
    /// Ping message.
    Ping,
    /// Pong message.
    Pong,
}

/// WebSocket channel types for subscription.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WsChannel {
    /// Public trades.
    Trades(String),
    /// Public klines.
    Klines(String, String),
    /// Public incremental depth.
    DepthIncremental(String),
    /// Public limit depth.
    DepthLimit(String, u32),
    /// Mini ticker for symbol.
    MiniTicker(String),
    /// All mini tickers.
    AllMiniTickers,
    /// Book ticker for symbol.
    BookTicker(String),
    /// Private account updates.
    Account,
    /// Private order updates.
    Orders,
    /// Private trade updates.
    Trades24h,
}

impl WsChannel {
    /// Get the channel string for subscription.
    pub fn to_channel_string(&self) -> String {
        match self {
            WsChannel::Trades(symbol) => format!("spot@public.deals.v3.api@{symbol}"),
            WsChannel::Klines(symbol, interval) => {
                format!("spot@public.kline.v3.api@{symbol}@{interval}")
            }
            WsChannel::DepthIncremental(symbol) => {
                format!("spot@public.increase.depth.v3.api@{symbol}")
            }
            WsChannel::DepthLimit(symbol, levels) => {
                format!("spot@public.limit.depth.v3.api@{symbol}@{levels}")
            }
            WsChannel::MiniTicker(symbol) => format!("spot@public.miniTicker.v3.api@{symbol}"),
            WsChannel::AllMiniTickers => "spot@public.miniTickers.v3.api".to_string(),
            WsChannel::BookTicker(symbol) => format!("spot@public.bookTicker.v3.api@{symbol}"),
            WsChannel::Account => "spot@private.account.v3.api".to_string(),
            WsChannel::Orders => "spot@private.orders.v3.api".to_string(),
            WsChannel::Trades24h => "spot@private.deals.v3.api".to_string(),
        }
    }
}

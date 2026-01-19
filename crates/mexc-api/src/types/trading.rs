//! Trading types for orders and trades.

use super::common::{OrderSide, OrderStatus, OrderType, StringDecimal, TimeInForce};
use serde::{Deserialize, Serialize};

/// Order fill information.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderFill {
    /// Fill price.
    pub price: StringDecimal,
    /// Fill quantity.
    pub qty: StringDecimal,
    /// Commission amount.
    pub commission: StringDecimal,
    /// Commission asset.
    pub commission_asset: String,
    /// Trade ID.
    #[serde(default)]
    pub trade_id: Option<i64>,
}

/// New order response (ACK).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewOrderAck {
    /// Symbol.
    pub symbol: String,
    /// Order ID.
    pub order_id: String,
    /// Order list ID (-1 if not OCO).
    #[serde(default)]
    pub order_list_id: Option<i64>,
    /// Client order ID.
    #[serde(default)]
    pub client_order_id: Option<String>,
    /// Transaction time.
    pub transact_time: i64,
}

/// New order response (RESULT).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewOrderResult {
    /// Symbol.
    pub symbol: String,
    /// Order ID.
    pub order_id: String,
    /// Order list ID (-1 if not OCO).
    #[serde(default)]
    pub order_list_id: Option<i64>,
    /// Client order ID.
    #[serde(default)]
    pub client_order_id: Option<String>,
    /// Transaction time.
    pub transact_time: i64,
    /// Order price.
    pub price: StringDecimal,
    /// Original quantity.
    pub orig_qty: StringDecimal,
    /// Executed quantity.
    pub executed_qty: StringDecimal,
    /// Cumulative quote quantity.
    pub cummulative_quote_qty: StringDecimal,
    /// Order status.
    pub status: OrderStatus,
    /// Time in force.
    pub time_in_force: Option<TimeInForce>,
    /// Order type.
    #[serde(rename = "type")]
    pub order_type: OrderType,
    /// Order side.
    pub side: OrderSide,
}

/// New order response (FULL).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NewOrderFull {
    /// Symbol.
    pub symbol: String,
    /// Order ID.
    pub order_id: String,
    /// Order list ID (-1 if not OCO).
    #[serde(default)]
    pub order_list_id: Option<i64>,
    /// Client order ID.
    #[serde(default)]
    pub client_order_id: Option<String>,
    /// Transaction time.
    pub transact_time: i64,
    /// Order price.
    pub price: StringDecimal,
    /// Original quantity.
    pub orig_qty: StringDecimal,
    /// Executed quantity.
    pub executed_qty: StringDecimal,
    /// Cumulative quote quantity.
    pub cummulative_quote_qty: StringDecimal,
    /// Order status.
    pub status: OrderStatus,
    /// Time in force.
    pub time_in_force: Option<TimeInForce>,
    /// Order type.
    #[serde(rename = "type")]
    pub order_type: OrderType,
    /// Order side.
    pub side: OrderSide,
    /// Order fills.
    #[serde(default)]
    pub fills: Vec<OrderFill>,
}

/// Order query response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Order {
    /// Symbol.
    pub symbol: String,
    /// Order ID.
    pub order_id: String,
    /// Order list ID (-1 if not OCO).
    #[serde(default)]
    pub order_list_id: Option<i64>,
    /// Client order ID.
    #[serde(default)]
    pub client_order_id: Option<String>,
    /// Order price.
    pub price: StringDecimal,
    /// Original quantity.
    pub orig_qty: StringDecimal,
    /// Executed quantity.
    pub executed_qty: StringDecimal,
    /// Cumulative quote quantity.
    pub cummulative_quote_qty: StringDecimal,
    /// Order status.
    pub status: OrderStatus,
    /// Time in force.
    pub time_in_force: Option<TimeInForce>,
    /// Order type.
    #[serde(rename = "type")]
    pub order_type: OrderType,
    /// Order side.
    pub side: OrderSide,
    /// Stop price (for stop orders).
    #[serde(default)]
    pub stop_price: Option<StringDecimal>,
    /// Iceberg quantity.
    #[serde(default)]
    pub iceberg_qty: Option<StringDecimal>,
    /// Order creation time.
    pub time: i64,
    /// Last update time.
    pub update_time: i64,
    /// Is the order working?
    pub is_working: bool,
    /// Original quote order quantity.
    #[serde(default)]
    pub orig_quote_order_qty: Option<StringDecimal>,
}

/// Cancel order response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancelOrderResponse {
    /// Symbol.
    pub symbol: String,
    /// Order ID.
    pub order_id: String,
    /// Original client order ID.
    #[serde(default)]
    pub orig_client_order_id: Option<String>,
    /// Client order ID.
    #[serde(default)]
    pub client_order_id: Option<String>,
    /// Order price.
    pub price: StringDecimal,
    /// Original quantity.
    pub orig_qty: StringDecimal,
    /// Executed quantity.
    pub executed_qty: StringDecimal,
    /// Cumulative quote quantity.
    pub cummulative_quote_qty: StringDecimal,
    /// Order status.
    pub status: OrderStatus,
    /// Time in force.
    pub time_in_force: Option<TimeInForce>,
    /// Order type.
    #[serde(rename = "type")]
    pub order_type: OrderType,
    /// Order side.
    pub side: OrderSide,
}

/// Account trade.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountTrade {
    /// Symbol.
    pub symbol: String,
    /// Trade ID.
    pub id: i64,
    /// Order ID.
    pub order_id: String,
    /// Order list ID.
    #[serde(default)]
    pub order_list_id: Option<i64>,
    /// Trade price.
    pub price: StringDecimal,
    /// Trade quantity.
    pub qty: StringDecimal,
    /// Quote quantity.
    pub quote_qty: StringDecimal,
    /// Commission.
    pub commission: StringDecimal,
    /// Commission asset.
    pub commission_asset: String,
    /// Trade time.
    pub time: i64,
    /// Was buyer?
    pub is_buyer: bool,
    /// Was maker?
    pub is_maker: bool,
    /// Is best match?
    pub is_best_match: bool,
}

/// Batch order request item.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchOrderRequest {
    /// Symbol.
    pub symbol: String,
    /// Order side.
    pub side: OrderSide,
    /// Order type.
    #[serde(rename = "type")]
    pub order_type: OrderType,
    /// Quantity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quantity: Option<StringDecimal>,
    /// Quote order quantity (for market orders).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quote_order_qty: Option<StringDecimal>,
    /// Price (required for limit orders).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<StringDecimal>,
    /// Client order ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new_client_order_id: Option<String>,
}

/// Batch orders response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum BatchOrderResult {
    /// Successful order.
    Success(NewOrderResult),
    /// Failed order.
    Error {
        /// Error code.
        code: i32,
        /// Error message.
        msg: String,
    },
}

/// Trade fee information.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TradeFee {
    /// Symbol.
    pub symbol: String,
    /// Maker commission.
    pub maker_commission: StringDecimal,
    /// Taker commission.
    pub taker_commission: StringDecimal,
}

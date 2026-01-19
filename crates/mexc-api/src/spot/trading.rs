//! Trading endpoints for spot trading.

use std::collections::HashMap;

use crate::client::MexcClient;
use crate::error::Result;
use crate::rate_limit::EndpointWeight;
use crate::types::{
    AccountTrade, BatchOrderRequest, BatchOrderResult, CancelOrderResponse, NewOrderAck,
    NewOrderFull, NewOrderRespType, NewOrderResult, Order, OrderSide, OrderType, StringDecimal,
    TimeInForce, TradeFee,
};

/// Trading API.
#[derive(Debug, Clone)]
pub struct TradingApi {
    client: MexcClient,
}

impl TradingApi {
    /// Create a new Trading API instance.
    pub fn new(client: MexcClient) -> Self {
        Self { client }
    }

    /// Test new order (validates without placing).
    ///
    /// # Arguments
    /// * `symbol` - Trading pair
    /// * `side` - Order side (BUY/SELL)
    /// * `order_type` - Order type
    /// * `options` - Additional order options
    pub async fn test_order(
        &self,
        symbol: &str,
        side: OrderSide,
        order_type: OrderType,
        options: OrderOptions,
    ) -> Result<NewOrderAck> {
        let params = self.build_order_params(symbol, side, order_type, options);

        self.client
            .signed_post("/order/test", params, EndpointWeight::ORDER)
            .await
    }

    /// Place a new order.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair
    /// * `side` - Order side (BUY/SELL)
    /// * `order_type` - Order type
    /// * `options` - Additional order options
    ///
    /// # Example
    /// ```ignore
    /// use mexc_api::types::{OrderSide, OrderType};
    ///
    /// let order = trading.new_order(
    ///     "BTCUSDT",
    ///     OrderSide::Buy,
    ///     OrderType::Limit,
    ///     OrderOptions::new()
    ///         .quantity("0.001")
    ///         .price("50000")
    ///         .response_type(NewOrderRespType::Full),
    /// ).await?;
    /// ```
    pub async fn new_order(
        &self,
        symbol: &str,
        side: OrderSide,
        order_type: OrderType,
        options: OrderOptions,
    ) -> Result<NewOrderFull> {
        let params = self.build_order_params(symbol, side, order_type, options);

        self.client
            .signed_post("/order", params, EndpointWeight::ORDER)
            .await
    }

    /// Place a new order with ACK response type (minimal response).
    pub async fn new_order_ack(
        &self,
        symbol: &str,
        side: OrderSide,
        order_type: OrderType,
        options: OrderOptions,
    ) -> Result<NewOrderAck> {
        let mut options = options;
        options.new_order_resp_type = Some(NewOrderRespType::Ack);

        let params = self.build_order_params(symbol, side, order_type, options);

        self.client
            .signed_post("/order", params, EndpointWeight::ORDER)
            .await
    }

    /// Place a new order with RESULT response type.
    pub async fn new_order_result(
        &self,
        symbol: &str,
        side: OrderSide,
        order_type: OrderType,
        options: OrderOptions,
    ) -> Result<NewOrderResult> {
        let mut options = options;
        options.new_order_resp_type = Some(NewOrderRespType::Result);

        let params = self.build_order_params(symbol, side, order_type, options);

        self.client
            .signed_post("/order", params, EndpointWeight::ORDER)
            .await
    }

    /// Place a limit order (convenience method).
    pub async fn limit_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: &str,
        price: &str,
    ) -> Result<NewOrderFull> {
        self.new_order(
            symbol,
            side,
            OrderType::Limit,
            OrderOptions::new()
                .quantity(quantity)
                .price(price)
                .time_in_force(TimeInForce::Gtc),
        )
        .await
    }

    /// Place a market order (convenience method).
    pub async fn market_order(
        &self,
        symbol: &str,
        side: OrderSide,
        quantity: &str,
    ) -> Result<NewOrderFull> {
        self.new_order(
            symbol,
            side,
            OrderType::Market,
            OrderOptions::new().quantity(quantity),
        )
        .await
    }

    /// Place a market order by quote quantity (convenience method).
    pub async fn market_order_quote(
        &self,
        symbol: &str,
        side: OrderSide,
        quote_quantity: &str,
    ) -> Result<NewOrderFull> {
        self.new_order(
            symbol,
            side,
            OrderType::Market,
            OrderOptions::new().quote_order_qty(quote_quantity),
        )
        .await
    }

    /// Place batch orders.
    ///
    /// # Arguments
    /// * `orders` - List of orders to place (max 20)
    ///
    /// Note: Rate limited to 2 requests per second.
    pub async fn batch_orders(&self, orders: Vec<BatchOrderRequest>) -> Result<Vec<BatchOrderResult>> {
        let orders_json = serde_json::to_string(&orders)?;

        let mut params = HashMap::new();
        params.insert("batchOrders".to_string(), orders_json);

        self.client
            .signed_post("/batchOrders", params, EndpointWeight::ORDER)
            .await
    }

    /// Query an order by order ID.
    pub async fn get_order(&self, symbol: &str, order_id: &str) -> Result<Order> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());
        params.insert("orderId".to_string(), order_id.to_string());

        self.client
            .signed_get("/order", params, EndpointWeight::ORDER)
            .await
    }

    /// Query an order by client order ID.
    pub async fn get_order_by_client_id(
        &self,
        symbol: &str,
        client_order_id: &str,
    ) -> Result<Order> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());
        params.insert(
            "origClientOrderId".to_string(),
            client_order_id.to_string(),
        );

        self.client
            .signed_get("/order", params, EndpointWeight::ORDER)
            .await
    }

    /// Get all open orders.
    ///
    /// # Arguments
    /// * `symbol` - Optional trading pair (max 5 if specified)
    pub async fn open_orders(&self, symbol: Option<&str>) -> Result<Vec<Order>> {
        let mut params = HashMap::new();

        if let Some(s) = symbol {
            params.insert("symbol".to_string(), s.to_uppercase());
        }

        self.client
            .signed_get("/openOrders", params, EndpointWeight::ORDER)
            .await
    }

    /// Get all orders (open, canceled, filled).
    ///
    /// # Arguments
    /// * `symbol` - Trading pair
    /// * `order_id` - Filter from this order ID
    /// * `start_time` - Start timestamp
    /// * `end_time` - End timestamp
    /// * `limit` - Number of orders (default 500, max 1000)
    ///
    /// Note: Maximum query period is 7 days; default is 24 hours.
    pub async fn all_orders(
        &self,
        symbol: &str,
        order_id: Option<&str>,
        start_time: Option<i64>,
        end_time: Option<i64>,
        limit: Option<u32>,
    ) -> Result<Vec<Order>> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());

        if let Some(id) = order_id {
            params.insert("orderId".to_string(), id.to_string());
        }

        if let Some(ts) = start_time {
            params.insert("startTime".to_string(), ts.to_string());
        }

        if let Some(ts) = end_time {
            params.insert("endTime".to_string(), ts.to_string());
        }

        if let Some(l) = limit {
            params.insert("limit".to_string(), l.to_string());
        }

        self.client
            .signed_get("/allOrders", params, EndpointWeight::ORDER)
            .await
    }

    /// Cancel an order by order ID.
    pub async fn cancel_order(&self, symbol: &str, order_id: &str) -> Result<CancelOrderResponse> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());
        params.insert("orderId".to_string(), order_id.to_string());

        self.client
            .signed_delete("/order", params, EndpointWeight::ORDER)
            .await
    }

    /// Cancel an order by client order ID.
    pub async fn cancel_order_by_client_id(
        &self,
        symbol: &str,
        client_order_id: &str,
    ) -> Result<CancelOrderResponse> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());
        params.insert(
            "origClientOrderId".to_string(),
            client_order_id.to_string(),
        );

        self.client
            .signed_delete("/order", params, EndpointWeight::ORDER)
            .await
    }

    /// Cancel all open orders on a symbol.
    pub async fn cancel_all_orders(&self, symbol: &str) -> Result<Vec<CancelOrderResponse>> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());

        self.client
            .signed_delete("/openOrders", params, EndpointWeight::ORDER)
            .await
    }

    /// Get account trade history.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair
    /// * `order_id` - Filter by order ID
    /// * `start_time` - Start timestamp
    /// * `end_time` - End timestamp
    /// * `from_id` - Trade ID to fetch from
    /// * `limit` - Number of trades (default 500, max 1000)
    pub async fn my_trades(
        &self,
        symbol: &str,
        order_id: Option<&str>,
        start_time: Option<i64>,
        end_time: Option<i64>,
        from_id: Option<i64>,
        limit: Option<u32>,
    ) -> Result<Vec<AccountTrade>> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());

        if let Some(id) = order_id {
            params.insert("orderId".to_string(), id.to_string());
        }

        if let Some(ts) = start_time {
            params.insert("startTime".to_string(), ts.to_string());
        }

        if let Some(ts) = end_time {
            params.insert("endTime".to_string(), ts.to_string());
        }

        if let Some(id) = from_id {
            params.insert("fromId".to_string(), id.to_string());
        }

        if let Some(l) = limit {
            params.insert("limit".to_string(), l.to_string());
        }

        self.client
            .signed_get("/myTrades", params, EndpointWeight::MY_TRADES)
            .await
    }

    /// Get trading fee rates.
    ///
    /// # Arguments
    /// * `symbol` - Optional trading pair
    pub async fn trade_fee(&self, symbol: Option<&str>) -> Result<Vec<TradeFee>> {
        let mut params = HashMap::new();

        if let Some(s) = symbol {
            params.insert("symbol".to_string(), s.to_uppercase());
        }

        self.client
            .signed_get("/tradeFee", params, EndpointWeight::TRADE_FEE)
            .await
    }

    /// Build order parameters.
    fn build_order_params(
        &self,
        symbol: &str,
        side: OrderSide,
        order_type: OrderType,
        options: OrderOptions,
    ) -> HashMap<String, String> {
        let mut params = HashMap::new();

        params.insert("symbol".to_string(), symbol.to_uppercase());
        params.insert("side".to_string(), side.to_string());
        params.insert("type".to_string(), order_type.to_string());

        if let Some(qty) = options.quantity {
            params.insert("quantity".to_string(), qty.to_string());
        }

        if let Some(qty) = options.quote_order_qty {
            params.insert("quoteOrderQty".to_string(), qty.to_string());
        }

        if let Some(price) = options.price {
            params.insert("price".to_string(), price.to_string());
        }

        if let Some(tif) = options.time_in_force {
            params.insert("timeInForce".to_string(), tif.to_string());
        }

        if let Some(id) = options.new_client_order_id {
            params.insert("newClientOrderId".to_string(), id);
        }

        if let Some(price) = options.stop_price {
            params.insert("stopPrice".to_string(), price.to_string());
        }

        if let Some(qty) = options.iceberg_qty {
            params.insert("icebergQty".to_string(), qty.to_string());
        }

        if let Some(resp_type) = options.new_order_resp_type {
            params.insert("newOrderRespType".to_string(), resp_type.to_string());
        }

        params
    }
}

/// Options for placing an order.
#[derive(Debug, Clone, Default)]
pub struct OrderOptions {
    /// Order quantity.
    pub quantity: Option<StringDecimal>,
    /// Quote order quantity (for market orders).
    pub quote_order_qty: Option<StringDecimal>,
    /// Order price (required for limit orders).
    pub price: Option<StringDecimal>,
    /// Time in force.
    pub time_in_force: Option<TimeInForce>,
    /// Client order ID.
    pub new_client_order_id: Option<String>,
    /// Stop price.
    pub stop_price: Option<StringDecimal>,
    /// Iceberg quantity.
    pub iceberg_qty: Option<StringDecimal>,
    /// Response type.
    pub new_order_resp_type: Option<NewOrderRespType>,
}

impl OrderOptions {
    /// Create new order options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the quantity.
    pub fn quantity(mut self, qty: &str) -> Self {
        self.quantity = qty.parse().ok().map(StringDecimal::new);
        self
    }

    /// Set the quote order quantity.
    pub fn quote_order_qty(mut self, qty: &str) -> Self {
        self.quote_order_qty = qty.parse().ok().map(StringDecimal::new);
        self
    }

    /// Set the price.
    pub fn price(mut self, price: &str) -> Self {
        self.price = price.parse().ok().map(StringDecimal::new);
        self
    }

    /// Set the time in force.
    pub fn time_in_force(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = Some(tif);
        self
    }

    /// Set the client order ID.
    pub fn client_order_id(mut self, id: impl Into<String>) -> Self {
        self.new_client_order_id = Some(id.into());
        self
    }

    /// Set the stop price.
    pub fn stop_price(mut self, price: &str) -> Self {
        self.stop_price = price.parse().ok().map(StringDecimal::new);
        self
    }

    /// Set the iceberg quantity.
    pub fn iceberg_qty(mut self, qty: &str) -> Self {
        self.iceberg_qty = qty.parse().ok().map(StringDecimal::new);
        self
    }

    /// Set the response type.
    pub fn response_type(mut self, resp_type: NewOrderRespType) -> Self {
        self.new_order_resp_type = Some(resp_type);
        self
    }
}

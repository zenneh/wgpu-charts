//! Market data endpoints for spot trading.

use std::collections::HashMap;

use crate::client::MexcClient;
use crate::error::Result;
use crate::rate_limit::EndpointWeight;
use crate::types::{
    AggTrade, AvgPrice, BookTicker, Empty, ExchangeInfo, Kline, KlineInterval, OrderBook,
    PriceTicker, ServerTime, Ticker24hr, Trade,
};

/// Market data API.
#[derive(Debug, Clone)]
pub struct MarketApi {
    client: MexcClient,
}

impl MarketApi {
    /// Create a new Market API instance.
    pub fn new(client: MexcClient) -> Self {
        Self { client }
    }

    /// Test connectivity to the API.
    ///
    /// # Example
    /// ```ignore
    /// let client = MexcClient::public()?;
    /// let market = MarketApi::new(client);
    /// market.ping().await?;
    /// ```
    pub async fn ping(&self) -> Result<Empty> {
        self.client.get("/ping", None, EndpointWeight::PING).await
    }

    /// Get server time.
    ///
    /// # Example
    /// ```ignore
    /// let time = market.time().await?;
    /// println!("Server time: {}", time.server_time);
    /// ```
    pub async fn time(&self) -> Result<ServerTime> {
        self.client.get("/time", None, EndpointWeight::TIME).await
    }

    /// Get exchange information.
    ///
    /// Returns trading rules and symbol information.
    ///
    /// # Arguments
    /// * `symbol` - Optional symbol to filter (e.g., "BTCUSDT")
    /// * `symbols` - Optional list of symbols to filter
    pub async fn exchange_info(
        &self,
        symbol: Option<&str>,
        symbols: Option<&[&str]>,
    ) -> Result<ExchangeInfo> {
        let mut params = HashMap::new();

        if let Some(s) = symbol {
            params.insert("symbol".to_string(), s.to_uppercase());
        }

        if let Some(syms) = symbols {
            let syms_str = syms
                .iter()
                .map(|s| s.to_uppercase())
                .collect::<Vec<_>>()
                .join(",");
            params.insert("symbols".to_string(), syms_str);
        }

        let params = if params.is_empty() {
            None
        } else {
            Some(params)
        };

        self.client
            .get("/exchangeInfo", params, EndpointWeight::EXCHANGE_INFO)
            .await
    }

    /// Get order book depth.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSDT")
    /// * `limit` - Depth limit (valid: 5, 10, 20, 50, 100, 500, 1000, 5000)
    pub async fn depth(&self, symbol: &str, limit: Option<u32>) -> Result<OrderBook> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());

        if let Some(l) = limit {
            params.insert("limit".to_string(), l.to_string());
        }

        self.client
            .get("/depth", Some(params), EndpointWeight::DEPTH)
            .await
    }

    /// Get recent trades.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair
    /// * `limit` - Number of trades (default 500, max 1000)
    pub async fn trades(&self, symbol: &str, limit: Option<u32>) -> Result<Vec<Trade>> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());

        if let Some(l) = limit {
            params.insert("limit".to_string(), l.to_string());
        }

        self.client
            .get("/trades", Some(params), EndpointWeight::TRADES)
            .await
    }

    /// Get historical trades.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair
    /// * `limit` - Number of trades (default 500, max 1000)
    /// * `from_id` - Trade ID to fetch from
    pub async fn historical_trades(
        &self,
        symbol: &str,
        limit: Option<u32>,
        from_id: Option<i64>,
    ) -> Result<Vec<Trade>> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());

        if let Some(l) = limit {
            params.insert("limit".to_string(), l.to_string());
        }

        if let Some(id) = from_id {
            params.insert("fromId".to_string(), id.to_string());
        }

        self.client
            .get("/historicalTrades", Some(params), EndpointWeight::TRADES)
            .await
    }

    /// Get compressed/aggregate trades.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair
    /// * `from_id` - ID to get aggregate trades from (inclusive)
    /// * `start_time` - Start timestamp in milliseconds
    /// * `end_time` - End timestamp in milliseconds
    /// * `limit` - Number of trades (default 500, max 1000)
    ///
    /// Note: If sending start_time and end_time, interval must be less than 1 hour.
    pub async fn agg_trades(
        &self,
        symbol: &str,
        from_id: Option<i64>,
        start_time: Option<i64>,
        end_time: Option<i64>,
        limit: Option<u32>,
    ) -> Result<Vec<AggTrade>> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());

        if let Some(id) = from_id {
            params.insert("fromId".to_string(), id.to_string());
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
            .get("/aggTrades", Some(params), EndpointWeight::AGG_TRADES)
            .await
    }

    /// Get kline/candlestick data.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair
    /// * `interval` - Kline interval
    /// * `start_time` - Start timestamp in milliseconds
    /// * `end_time` - End timestamp in milliseconds
    /// * `limit` - Number of klines (default 500, max 1000)
    pub async fn klines(
        &self,
        symbol: &str,
        interval: KlineInterval,
        start_time: Option<i64>,
        end_time: Option<i64>,
        limit: Option<u32>,
    ) -> Result<Vec<Kline>> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());
        params.insert("interval".to_string(), interval.to_string());

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
            .get("/klines", Some(params), EndpointWeight::KLINES)
            .await
    }

    /// Get current average price.
    ///
    /// # Arguments
    /// * `symbol` - Trading pair
    pub async fn avg_price(&self, symbol: &str) -> Result<AvgPrice> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());

        self.client
            .get("/avgPrice", Some(params), EndpointWeight::AVG_PRICE)
            .await
    }

    /// Get 24hr ticker statistics.
    ///
    /// # Arguments
    /// * `symbol` - Optional trading pair (returns all if not specified)
    pub async fn ticker_24hr(&self, symbol: Option<&str>) -> Result<Vec<Ticker24hr>> {
        let (params, weight) = if let Some(s) = symbol {
            let mut params = HashMap::new();
            params.insert("symbol".to_string(), s.to_uppercase());
            (Some(params), EndpointWeight::TICKER_24HR_SINGLE)
        } else {
            (None, EndpointWeight::TICKER_24HR_ALL)
        };

        // Single symbol returns object, multiple returns array
        if symbol.is_some() {
            let ticker: Ticker24hr = self.client.get("/ticker/24hr", params, weight).await?;
            Ok(vec![ticker])
        } else {
            self.client.get("/ticker/24hr", params, weight).await
        }
    }

    /// Get 24hr ticker for a single symbol.
    pub async fn ticker_24hr_single(&self, symbol: &str) -> Result<Ticker24hr> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());

        self.client
            .get("/ticker/24hr", Some(params), EndpointWeight::TICKER_24HR_SINGLE)
            .await
    }

    /// Get price ticker.
    ///
    /// # Arguments
    /// * `symbol` - Optional trading pair (returns all if not specified)
    pub async fn ticker_price(&self, symbol: Option<&str>) -> Result<Vec<PriceTicker>> {
        let (params, weight) = if let Some(s) = symbol {
            let mut params = HashMap::new();
            params.insert("symbol".to_string(), s.to_uppercase());
            (Some(params), EndpointWeight::PRICE_SINGLE)
        } else {
            (None, EndpointWeight::PRICE_ALL)
        };

        // Single symbol returns object, multiple returns array
        if symbol.is_some() {
            let ticker: PriceTicker = self.client.get("/ticker/price", params, weight).await?;
            Ok(vec![ticker])
        } else {
            self.client.get("/ticker/price", params, weight).await
        }
    }

    /// Get price ticker for a single symbol.
    pub async fn ticker_price_single(&self, symbol: &str) -> Result<PriceTicker> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());

        self.client
            .get("/ticker/price", Some(params), EndpointWeight::PRICE_SINGLE)
            .await
    }

    /// Get book ticker (best bid/ask).
    ///
    /// # Arguments
    /// * `symbol` - Optional trading pair (returns all if not specified)
    pub async fn book_ticker(&self, symbol: Option<&str>) -> Result<Vec<BookTicker>> {
        let params = if let Some(s) = symbol {
            let mut params = HashMap::new();
            params.insert("symbol".to_string(), s.to_uppercase());
            Some(params)
        } else {
            None
        };

        // Single symbol returns object, multiple returns array
        if symbol.is_some() {
            let ticker: BookTicker = self
                .client
                .get("/ticker/bookTicker", params, EndpointWeight::BOOK_TICKER)
                .await?;
            Ok(vec![ticker])
        } else {
            self.client
                .get("/ticker/bookTicker", params, EndpointWeight::BOOK_TICKER)
                .await
        }
    }

    /// Get book ticker for a single symbol.
    pub async fn book_ticker_single(&self, symbol: &str) -> Result<BookTicker> {
        let mut params = HashMap::new();
        params.insert("symbol".to_string(), symbol.to_uppercase());

        self.client
            .get("/ticker/bookTicker", Some(params), EndpointWeight::BOOK_TICKER)
            .await
    }
}

#[cfg(test)]
mod tests {
    // Integration tests would go here
    // These would require a test API key or mock server
}

//! Spot trading API endpoints.

mod market;
mod trading;
mod account;

pub use market::*;
pub use trading::*;
pub use account::*;

use crate::client::MexcClient;

/// Spot API client wrapper.
#[derive(Debug, Clone)]
pub struct SpotApi {
    client: MexcClient,
}

impl SpotApi {
    /// Create a new Spot API client.
    pub fn new(client: MexcClient) -> Self {
        Self { client }
    }

    /// Get the underlying HTTP client.
    pub fn client(&self) -> &MexcClient {
        &self.client
    }

    /// Get market data API.
    pub fn market(&self) -> MarketApi {
        MarketApi::new(self.client.clone())
    }

    /// Get trading API.
    pub fn trading(&self) -> TradingApi {
        TradingApi::new(self.client.clone())
    }

    /// Get account API.
    pub fn account(&self) -> AccountApi {
        AccountApi::new(self.client.clone())
    }
}

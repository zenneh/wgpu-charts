//! Account endpoints for spot trading.

use std::collections::HashMap;

use crate::client::MexcClient;
use crate::error::Result;
use crate::rate_limit::EndpointWeight;
use crate::types::{
    AccountInfo, AccountType, CurrencyConfig, DepositAddress, DepositRecord, KycStatus,
    ListenKey, MxDeductStatus, TransferHistory, TransferResponse, WithdrawRecord, WithdrawResponse,
};

/// Account API.
#[derive(Debug, Clone)]
pub struct AccountApi {
    client: MexcClient,
}

impl AccountApi {
    /// Create a new Account API instance.
    pub fn new(client: MexcClient) -> Self {
        Self { client }
    }

    /// Get account information including balances.
    pub async fn info(&self) -> Result<AccountInfo> {
        self.client
            .signed_get("/account", HashMap::new(), EndpointWeight::ACCOUNT)
            .await
    }

    /// Get KYC verification status.
    pub async fn kyc_status(&self) -> Result<KycStatus> {
        self.client
            .signed_get("/kyc/status", HashMap::new(), EndpointWeight::ORDER)
            .await
    }

    /// Get all currency configurations.
    pub async fn currency_config(&self) -> Result<Vec<CurrencyConfig>> {
        self.client
            .signed_get(
                "/capital/config/getall",
                HashMap::new(),
                EndpointWeight::CURRENCY_CONFIG,
            )
            .await
    }

    /// Get deposit address for a coin.
    ///
    /// # Arguments
    /// * `coin` - Coin name (e.g., "BTC")
    /// * `network` - Optional network name
    pub async fn deposit_address(
        &self,
        coin: &str,
        network: Option<&str>,
    ) -> Result<DepositAddress> {
        let mut params = HashMap::new();
        params.insert("coin".to_string(), coin.to_uppercase());

        if let Some(net) = network {
            params.insert("network".to_string(), net.to_string());
        }

        self.client
            .signed_get(
                "/capital/deposit/address",
                params,
                EndpointWeight::DEPOSIT_ADDRESS,
            )
            .await
    }

    /// Get deposit history.
    ///
    /// # Arguments
    /// * `coin` - Optional coin filter
    /// * `status` - Optional status filter (0=pending, 1=success, 2=failed)
    /// * `start_time` - Start timestamp
    /// * `end_time` - End timestamp
    /// * `offset` - Pagination offset
    /// * `limit` - Number of records (default 1000, max 1000)
    pub async fn deposit_history(
        &self,
        coin: Option<&str>,
        status: Option<i32>,
        start_time: Option<i64>,
        end_time: Option<i64>,
        offset: Option<u32>,
        limit: Option<u32>,
    ) -> Result<Vec<DepositRecord>> {
        let mut params = HashMap::new();

        if let Some(c) = coin {
            params.insert("coin".to_string(), c.to_uppercase());
        }

        if let Some(s) = status {
            params.insert("status".to_string(), s.to_string());
        }

        if let Some(ts) = start_time {
            params.insert("startTime".to_string(), ts.to_string());
        }

        if let Some(ts) = end_time {
            params.insert("endTime".to_string(), ts.to_string());
        }

        if let Some(o) = offset {
            params.insert("offset".to_string(), o.to_string());
        }

        if let Some(l) = limit {
            params.insert("limit".to_string(), l.to_string());
        }

        self.client
            .signed_get("/capital/deposit/hisrec", params, EndpointWeight::ORDER)
            .await
    }

    /// Submit a withdrawal request.
    ///
    /// # Arguments
    /// * `coin` - Coin name
    /// * `network` - Network name
    /// * `address` - Withdrawal address
    /// * `amount` - Withdrawal amount
    /// * `address_tag` - Optional tag/memo
    /// * `withdraw_order_id` - Optional client withdrawal ID
    pub async fn withdraw(
        &self,
        coin: &str,
        network: &str,
        address: &str,
        amount: &str,
        address_tag: Option<&str>,
        withdraw_order_id: Option<&str>,
    ) -> Result<WithdrawResponse> {
        let mut params = HashMap::new();
        params.insert("coin".to_string(), coin.to_uppercase());
        params.insert("network".to_string(), network.to_string());
        params.insert("address".to_string(), address.to_string());
        params.insert("amount".to_string(), amount.to_string());

        if let Some(tag) = address_tag {
            params.insert("addressTag".to_string(), tag.to_string());
        }

        if let Some(id) = withdraw_order_id {
            params.insert("withdrawOrderId".to_string(), id.to_string());
        }

        self.client
            .signed_post("/capital/withdraw/apply", params, EndpointWeight::ORDER)
            .await
    }

    /// Cancel a withdrawal request.
    pub async fn cancel_withdraw(&self, id: &str) -> Result<()> {
        let mut params = HashMap::new();
        params.insert("id".to_string(), id.to_string());

        self.client
            .signed_delete::<serde_json::Value>("/capital/withdraw", params, EndpointWeight::ORDER)
            .await?;

        Ok(())
    }

    /// Get withdrawal history.
    ///
    /// # Arguments
    /// * `coin` - Optional coin filter
    /// * `withdraw_order_id` - Optional client withdrawal ID filter
    /// * `status` - Optional status filter
    /// * `start_time` - Start timestamp
    /// * `end_time` - End timestamp
    /// * `offset` - Pagination offset
    /// * `limit` - Number of records (default 1000, max 1000)
    pub async fn withdraw_history(
        &self,
        coin: Option<&str>,
        withdraw_order_id: Option<&str>,
        status: Option<i32>,
        start_time: Option<i64>,
        end_time: Option<i64>,
        offset: Option<u32>,
        limit: Option<u32>,
    ) -> Result<Vec<WithdrawRecord>> {
        let mut params = HashMap::new();

        if let Some(c) = coin {
            params.insert("coin".to_string(), c.to_uppercase());
        }

        if let Some(id) = withdraw_order_id {
            params.insert("withdrawOrderId".to_string(), id.to_string());
        }

        if let Some(s) = status {
            params.insert("status".to_string(), s.to_string());
        }

        if let Some(ts) = start_time {
            params.insert("startTime".to_string(), ts.to_string());
        }

        if let Some(ts) = end_time {
            params.insert("endTime".to_string(), ts.to_string());
        }

        if let Some(o) = offset {
            params.insert("offset".to_string(), o.to_string());
        }

        if let Some(l) = limit {
            params.insert("limit".to_string(), l.to_string());
        }

        self.client
            .signed_get("/capital/withdraw/history", params, EndpointWeight::ORDER)
            .await
    }

    /// Transfer between accounts (spot, futures, margin).
    ///
    /// # Arguments
    /// * `from_account` - Source account type
    /// * `to_account` - Destination account type
    /// * `asset` - Asset to transfer
    /// * `amount` - Amount to transfer
    pub async fn transfer(
        &self,
        from_account: AccountType,
        to_account: AccountType,
        asset: &str,
        amount: &str,
    ) -> Result<TransferResponse> {
        let mut params = HashMap::new();
        params.insert("fromAccountType".to_string(), from_account.to_string());
        params.insert("toAccountType".to_string(), to_account.to_string());
        params.insert("asset".to_string(), asset.to_uppercase());
        params.insert("amount".to_string(), amount.to_string());

        self.client
            .signed_post("/capital/transfer", params, EndpointWeight::ORDER)
            .await
    }

    /// Get transfer history.
    ///
    /// # Arguments
    /// * `from_account` - Source account type
    /// * `to_account` - Destination account type
    /// * `start_time` - Start timestamp
    /// * `end_time` - End timestamp
    /// * `page` - Page number
    /// * `size` - Page size
    pub async fn transfer_history(
        &self,
        from_account: AccountType,
        to_account: AccountType,
        start_time: Option<i64>,
        end_time: Option<i64>,
        page: Option<u32>,
        size: Option<u32>,
    ) -> Result<TransferHistory> {
        let mut params = HashMap::new();
        params.insert("fromAccountType".to_string(), from_account.to_string());
        params.insert("toAccountType".to_string(), to_account.to_string());

        if let Some(ts) = start_time {
            params.insert("startTime".to_string(), ts.to_string());
        }

        if let Some(ts) = end_time {
            params.insert("endTime".to_string(), ts.to_string());
        }

        if let Some(p) = page {
            params.insert("page".to_string(), p.to_string());
        }

        if let Some(s) = size {
            params.insert("size".to_string(), s.to_string());
        }

        self.client
            .signed_get("/capital/transfer", params, EndpointWeight::ORDER)
            .await
    }

    /// Get MX deduction status.
    pub async fn mx_deduct_status(&self) -> Result<MxDeductStatus> {
        self.client
            .signed_get("/mxDeduct/enable", HashMap::new(), EndpointWeight::ORDER)
            .await
    }

    /// Enable or disable MX deduction for fees.
    pub async fn set_mx_deduct(&self, enable: bool) -> Result<MxDeductStatus> {
        let mut params = HashMap::new();
        params.insert("mxDeductEnable".to_string(), enable.to_string());

        self.client
            .signed_post("/mxDeduct/enable", params, EndpointWeight::ORDER)
            .await
    }

    /// Create a listen key for user data stream.
    pub async fn create_listen_key(&self) -> Result<ListenKey> {
        self.client
            .signed_post("/userDataStream", HashMap::new(), EndpointWeight::ORDER)
            .await
    }

    /// Extend a listen key validity (by 60 minutes).
    pub async fn extend_listen_key(&self, listen_key: &str) -> Result<()> {
        let mut params = HashMap::new();
        params.insert("listenKey".to_string(), listen_key.to_string());

        self.client
            .signed_request::<serde_json::Value>(
                reqwest::Method::PUT,
                "/userDataStream",
                params,
                EndpointWeight::ORDER,
            )
            .await?;

        Ok(())
    }

    /// Close a listen key.
    pub async fn close_listen_key(&self, listen_key: &str) -> Result<()> {
        let mut params = HashMap::new();
        params.insert("listenKey".to_string(), listen_key.to_string());

        self.client
            .signed_delete::<serde_json::Value>("/userDataStream", params, EndpointWeight::ORDER)
            .await?;

        Ok(())
    }
}

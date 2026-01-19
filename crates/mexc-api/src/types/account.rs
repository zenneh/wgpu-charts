//! Account types.

use super::common::StringDecimal;
use serde::{Deserialize, Serialize};

/// Account information response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountInfo {
    /// Maker commission (in basis points).
    pub maker_commission: i32,
    /// Taker commission (in basis points).
    pub taker_commission: i32,
    /// Buyer commission.
    pub buyer_commission: i32,
    /// Seller commission.
    pub seller_commission: i32,
    /// Can trade.
    pub can_trade: bool,
    /// Can withdraw.
    pub can_withdraw: bool,
    /// Can deposit.
    pub can_deposit: bool,
    /// Last update time.
    #[serde(default)]
    pub update_time: Option<i64>,
    /// Account type.
    pub account_type: String,
    /// Account balances.
    pub balances: Vec<Balance>,
    /// Permissions.
    #[serde(default)]
    pub permissions: Vec<String>,
}

/// Account balance.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Balance {
    /// Asset name.
    pub asset: String,
    /// Free balance.
    pub free: StringDecimal,
    /// Locked balance.
    pub locked: StringDecimal,
}

impl Balance {
    /// Get total balance (free + locked).
    pub fn total(&self) -> rust_decimal::Decimal {
        self.free.0 + self.locked.0
    }
}

/// KYC status response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KycStatus {
    /// KYC level (1=Unverified, 2=Primary, 3=Advanced, 4=Institutional).
    pub level: i32,
}

impl KycStatus {
    /// Check if unverified.
    pub fn is_unverified(&self) -> bool {
        self.level == 1
    }

    /// Check if primary verified.
    pub fn is_primary(&self) -> bool {
        self.level == 2
    }

    /// Check if advanced verified.
    pub fn is_advanced(&self) -> bool {
        self.level == 3
    }

    /// Check if institutional.
    pub fn is_institutional(&self) -> bool {
        self.level == 4
    }
}

/// Currency configuration.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrencyConfig {
    /// Coin name.
    pub coin: String,
    /// Display name.
    #[serde(default)]
    pub name: Option<String>,
    /// Network list.
    pub network_list: Vec<NetworkConfig>,
}

/// Network configuration for a currency.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkConfig {
    /// Coin name.
    pub coin: String,
    /// Network name.
    pub network: String,
    /// Is default network.
    #[serde(default)]
    pub is_default: bool,
    /// Deposit enabled.
    pub deposit_enable: bool,
    /// Withdraw enabled.
    pub withdraw_enable: bool,
    /// Withdraw fee.
    #[serde(default)]
    pub withdraw_fee: Option<StringDecimal>,
    /// Minimum withdraw amount.
    #[serde(default)]
    pub withdraw_min: Option<StringDecimal>,
    /// Maximum withdraw amount.
    #[serde(default)]
    pub withdraw_max: Option<StringDecimal>,
    /// Minimum confirmations for deposit.
    #[serde(default)]
    pub min_confirm: Option<i32>,
    /// Confirmations to unlock.
    #[serde(default)]
    pub un_lock_confirm: Option<i32>,
    /// Network memo/tag regex.
    #[serde(default)]
    pub memo_regex: Option<String>,
    /// Contract address.
    #[serde(default)]
    pub contract_address: Option<String>,
}

/// Deposit address response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DepositAddress {
    /// Coin name.
    pub coin: String,
    /// Network name.
    pub network: String,
    /// Deposit address.
    pub address: String,
    /// Tag/memo (if required).
    #[serde(default)]
    pub tag: Option<String>,
}

/// Deposit history record.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DepositRecord {
    /// Deposit ID.
    pub id: String,
    /// Amount.
    pub amount: StringDecimal,
    /// Coin name.
    pub coin: String,
    /// Network.
    pub network: String,
    /// Status (0=pending, 1=success, 2=failed).
    pub status: i32,
    /// Deposit address.
    pub address: String,
    /// Tag/memo.
    #[serde(default)]
    pub tag: Option<String>,
    /// Transaction ID.
    pub tx_id: String,
    /// Insert time.
    pub insert_time: i64,
    /// Confirmations.
    #[serde(default)]
    pub confirm_times: Option<String>,
    /// Unlock confirmations.
    #[serde(default)]
    pub unlock_confirm: Option<i32>,
}

/// Withdraw history record.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WithdrawRecord {
    /// Withdraw ID.
    pub id: String,
    /// Amount.
    pub amount: StringDecimal,
    /// Coin name.
    pub coin: String,
    /// Network.
    pub network: String,
    /// Status.
    pub status: i32,
    /// Withdraw address.
    pub address: String,
    /// Tag/memo.
    #[serde(default)]
    pub tag: Option<String>,
    /// Transaction ID.
    #[serde(default)]
    pub tx_id: Option<String>,
    /// Apply time.
    pub apply_time: i64,
    /// Transaction fee.
    #[serde(default)]
    pub transaction_fee: Option<StringDecimal>,
    /// Client withdraw order ID.
    #[serde(default)]
    pub withdraw_order_id: Option<String>,
    /// Confirm number.
    #[serde(default)]
    pub confirm_no: Option<i32>,
}

/// Withdraw response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WithdrawResponse {
    /// Withdraw ID.
    pub id: String,
}

/// Transfer response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TransferResponse {
    /// Transfer transaction ID.
    pub tran_id: String,
}

/// Transfer record.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TransferRecord {
    /// Asset.
    pub asset: String,
    /// Amount.
    pub amount: StringDecimal,
    /// Transfer type.
    #[serde(rename = "type")]
    pub transfer_type: String,
    /// Status.
    pub status: String,
    /// Transaction ID.
    pub tran_id: String,
    /// Timestamp.
    pub timestamp: i64,
}

/// Transfer history response.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TransferHistory {
    /// Total records.
    pub total: i32,
    /// Transfer records.
    pub rows: Vec<TransferRecord>,
}

/// Listen key response.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ListenKey {
    /// Listen key for user data stream.
    pub listen_key: String,
}

/// Account type for transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum AccountType {
    /// Spot account.
    Spot,
    /// Futures account.
    Futures,
    /// Margin account.
    Margin,
}

impl std::fmt::Display for AccountType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AccountType::Spot => write!(f, "SPOT"),
            AccountType::Futures => write!(f, "FUTURES"),
            AccountType::Margin => write!(f, "MARGIN"),
        }
    }
}

/// MX deduction status.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MxDeductStatus {
    /// Is MX deduction enabled.
    pub mx_deduct_enable: bool,
}

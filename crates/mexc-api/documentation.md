# MEXC API Documentation

This document provides comprehensive documentation for the MEXC cryptocurrency exchange API, covering both Spot and Futures trading endpoints.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Rate Limits](#rate-limits)
4. [Spot API v3](#spot-api-v3)
   - [Market Data Endpoints](#market-data-endpoints-public)
   - [Account Endpoints](#account-endpoints-authenticated)
   - [Trading Endpoints](#trading-endpoints-authenticated)
   - [Wallet Endpoints](#wallet-endpoints-authenticated)
   - [Sub-Account Endpoints](#sub-account-endpoints)
5. [Futures API v1](#futures-api-v1)
   - [Market Endpoints](#futures-market-endpoints-public)
   - [Account Endpoints](#futures-account-endpoints-authenticated)
   - [Position Endpoints](#position-endpoints)
   - [Order Endpoints](#order-endpoints)
   - [Trigger Orders](#trigger-orders)
   - [Stop Orders](#stop-orders)
6. [WebSocket API](#websocket-api)
   - [Spot WebSocket](#spot-websocket)
   - [Futures WebSocket](#futures-websocket)
7. [Data Types](#data-types)
8. [Error Codes](#error-codes)

---

## Overview

### Base URLs

| API Type | REST Base URL | WebSocket URL |
|----------|---------------|---------------|
| Spot | `https://api.mexc.com` | `wss://wbs.mexc.com/ws` |
| Futures | `https://contract.mexc.com` | `wss://contract.mexc.com/edge` |

### Request Format

- **GET requests**: Parameters sent as query string
- **POST/DELETE requests**: Parameters may be sent as:
  - Query string with `Content-Type: application/x-www-form-urlencoded`
  - Request body with `Content-Type: application/json`
- Parameters may be mixed between query string and body
- Parameters can be sent in any order

### Response Format

All responses are JSON formatted. Successful responses typically include:

```json
{
  "code": 0,
  "data": { ... }
}
```

Or for spot API:

```json
{
  "symbol": "BTCUSDT",
  "price": "50000.00"
}
```

---

## Authentication

### Spot API Authentication

**Required Headers:**
```
X-MEXC-APIKEY: <your-api-key>
Content-Type: application/json
```

**Signature Generation (HMAC-SHA256):**

1. Build query string from all parameters including `timestamp`
2. Create HMAC-SHA256 signature using your secret key
3. Append signature to query string (lowercase hex)

```
signature = HMAC_SHA256(secret_key, query_string)
```

**Required Parameters for Signed Endpoints:**
- `timestamp` - Current timestamp in milliseconds
- `signature` - HMAC-SHA256 signature
- `recvWindow` (optional) - Request validity window (max 60000ms, default 5000ms)

**Example (Python):**
```python
import hmac
import hashlib
import time

api_key = "your_api_key"
api_secret = "your_api_secret"
timestamp = int(time.time() * 1000)

params = f"symbol=BTCUSDT&timestamp={timestamp}"
signature = hmac.new(
    api_secret.encode('utf-8'),
    params.encode('utf-8'),
    hashlib.sha256
).hexdigest()

url = f"https://api.mexc.com/api/v3/order?{params}&signature={signature}"
```

### Futures API Authentication

**Required Headers:**
```
ApiKey: <your-api-key>
Request-Time: <timestamp-in-milliseconds>
Signature: <hmac-sha256-signature>
Content-Type: application/json
Recv-Window: <optional-validity-window>
```

**Signature Generation:**
```
signature = HMAC_SHA256(secret_key, api_key + timestamp + request_params)
```

---

## Rate Limits

### Spot API Rate Limits

| Limit Type | Limit | Window |
|------------|-------|--------|
| IP-based | 500 requests | 10 seconds |
| UID-based | 500 requests | 10 seconds |
| WebSocket messages | 100 messages | 1 second |
| Batch orders | 2 requests | 1 second |

**Weight System:**
- Each endpoint has an assigned weight
- Total weight consumed counted against limits
- Weight varies: single symbol = 1, multiple symbols = 2-40

### Futures API Rate Limits

| Endpoint Type | Limit |
|---------------|-------|
| Market data | 20 requests / 2 seconds |
| Account queries | 20 requests / 2 seconds |
| Order placement | 20 requests / 2 seconds |
| Batch orders | 1 request / 2 seconds |
| Batch query | 5 requests / 2 seconds |

---

## Spot API v3

### Market Data Endpoints (Public)

#### Test Connectivity
```
GET /api/v3/ping
```
**Weight:** 1
**Response:** `{}`

#### Server Time
```
GET /api/v3/time
```
**Weight:** 1
**Response:**
```json
{
  "serverTime": 1704067200000
}
```

#### Default Symbols
```
GET /api/v3/defaultSymbols
```
**Weight:** 1
**Response:** Array of default trading symbols

#### Exchange Information
```
GET /api/v3/exchangeInfo
```
**Weight:** 10
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Single symbol (e.g., "BTCUSDT") |
| symbols | STRING | No | Comma-separated symbols |

**Response:**
```json
{
  "timezone": "UTC",
  "serverTime": 1704067200000,
  "rateLimits": [...],
  "symbols": [
    {
      "symbol": "BTCUSDT",
      "status": "TRADING",
      "baseAsset": "BTC",
      "baseAssetPrecision": 8,
      "quoteAsset": "USDT",
      "quoteAssetPrecision": 8,
      "quotePrecision": 8,
      "orderTypes": ["LIMIT", "MARKET", "LIMIT_MAKER"],
      "isSpotTradingAllowed": true,
      "isMarginTradingAllowed": false,
      "permissions": ["SPOT"],
      "filters": [...],
      "baseSizePrecision": "0.00001",
      "maxQuoteAmount": "5000000",
      "makerCommission": "0.002",
      "takerCommission": "0.002"
    }
  ]
}
```

#### Order Book (Depth)
```
GET /api/v3/depth
```
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair |
| limit | INT | No | Default 100; max 5000. Valid: [5, 10, 20, 50, 100, 500, 1000, 5000] |

**Response:**
```json
{
  "lastUpdateId": 1234567890,
  "bids": [
    ["50000.00", "1.5"],    // [price, quantity]
    ["49999.00", "2.0"]
  ],
  "asks": [
    ["50001.00", "0.5"],
    ["50002.00", "1.0"]
  ]
}
```

#### Recent Trades
```
GET /api/v3/trades
```
**Weight:** 5
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair |
| limit | INT | No | Default 500; max 1000 |

**Response:**
```json
[
  {
    "id": 123456789,
    "price": "50000.00",
    "qty": "0.1",
    "quoteQty": "5000.00",
    "time": 1704067200000,
    "isBuyerMaker": true,
    "isBestMatch": true
  }
]
```

#### Historical Trades
```
GET /api/v3/historicalTrades
```
**Weight:** 5
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair |
| limit | INT | No | Default 500; max 1000 |
| fromId | LONG | No | Trade ID to fetch from |

#### Aggregate Trades
```
GET /api/v3/aggTrades
```
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair |
| fromId | LONG | No | ID to get trades from (inclusive) |
| startTime | LONG | No | Timestamp in ms (inclusive) |
| endTime | LONG | No | Timestamp in ms (inclusive) |
| limit | INT | No | Default 500; max 1000 |

**Note:** If sending startTime and endTime, interval must be less than 1 hour.

**Response:**
```json
[
  {
    "a": 123456789,        // Aggregate trade ID
    "p": "50000.00",       // Price
    "q": "0.1",            // Quantity
    "f": 100,              // First trade ID
    "l": 105,              // Last trade ID
    "T": 1704067200000,    // Timestamp
    "m": true,             // Was buyer maker?
    "M": true              // Was trade best price match?
  }
]
```

#### Kline/Candlestick Data
```
GET /api/v3/klines
```
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair |
| interval | ENUM | Yes | Kline interval |
| startTime | LONG | No | Start timestamp (ms) |
| endTime | LONG | No | End timestamp (ms) |
| limit | INT | No | Default 500; max 1000 |

**Kline Intervals:**
`1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `12h`, `1d`, `1w`, `1M`

**Response:**
```json
[
  [
    1704067200000,    // Open time
    "50000.00",       // Open price
    "50100.00",       // High price
    "49900.00",       // Low price
    "50050.00",       // Close price
    "100.5",          // Volume
    1704070800000,    // Close time
    "5025000.00",     // Quote asset volume
    150,              // Number of trades
    "60.5",           // Taker buy base volume
    "3030000.00",     // Taker buy quote volume
    "0"               // Ignore
  ]
]
```

#### Current Average Price
```
GET /api/v3/avgPrice
```
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair |

**Response:**
```json
{
  "mins": 5,
  "price": "50000.00"
}
```

#### 24hr Ticker Statistics
```
GET /api/v3/ticker/24hr
```
**Weight:** 1 (single symbol), 40 (all symbols)
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Trading pair (optional) |

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "priceChange": "500.00",
  "priceChangePercent": "1.00",
  "prevClosePrice": "49500.00",
  "lastPrice": "50000.00",
  "bidPrice": "49999.00",
  "bidQty": "1.5",
  "askPrice": "50001.00",
  "askQty": "0.5",
  "openPrice": "49500.00",
  "highPrice": "50500.00",
  "lowPrice": "49000.00",
  "volume": "10000.00",
  "quoteVolume": "500000000.00",
  "openTime": 1703980800000,
  "closeTime": 1704067200000,
  "count": 150000
}
```

#### Price Ticker
```
GET /api/v3/ticker/price
```
**Weight:** 1 (single), 2 (all)
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Trading pair |

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "price": "50000.00"
}
```

#### Book Ticker
```
GET /api/v3/ticker/bookTicker
```
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Trading pair |

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "bidPrice": "49999.00",
  "bidQty": "1.5",
  "askPrice": "50001.00",
  "askQty": "0.5"
}
```

---

### Account Endpoints (Authenticated)

#### KYC Status
```
GET /api/v3/kyc/status
```
**Permission:** SPOT_ACCOUNT_READ
**Response:**
```json
{
  "level": 3
}
```
**Levels:** 1 = Unverified, 2 = Primary, 3 = Advanced, 4 = Institutional

#### Account Information
```
GET /api/v3/account
```
**Permission:** SPOT_ACCOUNT_READ
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| timestamp | LONG | Yes | Request timestamp |
| recvWindow | LONG | No | Valid window (max 60000) |

**Response:**
```json
{
  "makerCommission": 20,
  "takerCommission": 20,
  "buyerCommission": 0,
  "sellerCommission": 0,
  "canTrade": true,
  "canWithdraw": true,
  "canDeposit": true,
  "updateTime": null,
  "accountType": "SPOT",
  "balances": [
    {
      "asset": "BTC",
      "free": "1.5",
      "locked": "0.5"
    },
    {
      "asset": "USDT",
      "free": "10000.00",
      "locked": "5000.00"
    }
  ],
  "permissions": ["SPOT"]
}
```

#### User's Default Symbols
```
GET /api/v3/selfSymbols
```
**Permission:** SPOT_ACCOUNT_READ
**Weight:** 1

#### Account Trade List
```
GET /api/v3/myTrades
```
**Permission:** SPOT_ACCOUNT_READ
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair |
| orderId | LONG | No | Only with symbol |
| startTime | LONG | No | Start timestamp |
| endTime | LONG | No | End timestamp |
| fromId | LONG | No | Trade ID to fetch from |
| limit | INT | No | Default 500; max 1000 |
| timestamp | LONG | Yes | Request timestamp |

**Response:**
```json
[
  {
    "symbol": "BTCUSDT",
    "id": 123456789,
    "orderId": 987654321,
    "orderListId": -1,
    "price": "50000.00",
    "qty": "0.1",
    "quoteQty": "5000.00",
    "commission": "0.001",
    "commissionAsset": "BTC",
    "time": 1704067200000,
    "isBuyer": true,
    "isMaker": false,
    "isBestMatch": true
  }
]
```

#### Trade Fee
```
GET /api/v3/tradeFee
```
**Permission:** SPOT_ACCOUNT_READ
**Weight:** 20
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Trading pair |

**Response:**
```json
[
  {
    "symbol": "BTCUSDT",
    "makerCommission": "0.002",
    "takerCommission": "0.002"
  }
]
```

---

### Trading Endpoints (Authenticated)

#### Test New Order (No Execution)
```
POST /api/v3/order/test
```
**Permission:** SPOT_DEAL_WRITE
**Weight:** 1

Validates order parameters without placing the order.

#### New Order
```
POST /api/v3/order
```
**Permission:** SPOT_DEAL_WRITE
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair |
| side | ENUM | Yes | BUY or SELL |
| type | ENUM | Yes | Order type |
| timeInForce | ENUM | No | GTC, IOC, FOK |
| quantity | DECIMAL | No | Order quantity |
| quoteOrderQty | DECIMAL | No | Quote quantity (MARKET only) |
| price | DECIMAL | No | Order price |
| newClientOrderId | STRING | No | Unique order ID |
| stopPrice | DECIMAL | No | For stop orders |
| icebergQty | DECIMAL | No | For iceberg orders |
| newOrderRespType | ENUM | No | ACK, RESULT, or FULL |
| timestamp | LONG | Yes | Request timestamp |
| recvWindow | LONG | No | Valid window (max 60000) |

**Order Types:**
- `LIMIT` - Limit order (requires price, quantity, timeInForce)
- `MARKET` - Market order (requires quantity or quoteOrderQty)
- `LIMIT_MAKER` - Limit maker order (rejected if would immediately match)
- `IMMEDIATE_OR_CANCEL` (IOC) - Fill immediately or cancel remaining
- `FILL_OR_KILL` (FOK) - Fill entirely immediately or cancel

**Response (FULL):**
```json
{
  "symbol": "BTCUSDT",
  "orderId": 987654321,
  "orderListId": -1,
  "clientOrderId": "my_order_001",
  "transactTime": 1704067200000,
  "price": "50000.00",
  "origQty": "0.1",
  "executedQty": "0.1",
  "cummulativeQuoteQty": "5000.00",
  "status": "FILLED",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "fills": [
    {
      "price": "50000.00",
      "qty": "0.1",
      "commission": "0.0001",
      "commissionAsset": "BTC",
      "tradeId": 123456789
    }
  ]
}
```

#### Batch Orders
```
POST /api/v3/batchOrders
```
**Permission:** SPOT_DEAL_WRITE
**Weight:** 1
**Rate Limit:** 2 times/second
**Max Orders:** 20 per request

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| batchOrders | LIST | Yes | Array of order objects |
| timestamp | LONG | Yes | Request timestamp |

#### Query Order
```
GET /api/v3/order
```
**Permission:** SPOT_DEAL_READ
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair |
| orderId | LONG | No* | Order ID |
| origClientOrderId | STRING | No* | Client order ID |
| timestamp | LONG | Yes | Request timestamp |

*At least one of orderId or origClientOrderId required.

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "orderId": 987654321,
  "orderListId": -1,
  "clientOrderId": "my_order_001",
  "price": "50000.00",
  "origQty": "0.1",
  "executedQty": "0.05",
  "cummulativeQuoteQty": "2500.00",
  "status": "PARTIALLY_FILLED",
  "timeInForce": "GTC",
  "type": "LIMIT",
  "side": "BUY",
  "stopPrice": "0.00",
  "icebergQty": "0.00",
  "time": 1704067200000,
  "updateTime": 1704067300000,
  "isWorking": true,
  "origQuoteOrderQty": "0.00"
}
```

#### Current Open Orders
```
GET /api/v3/openOrders
```
**Permission:** SPOT_DEAL_READ
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Trading pair (up to 5 if specified) |
| timestamp | LONG | Yes | Request timestamp |

**Note:** Maximum 5 symbols can be queried at once.

#### All Orders
```
GET /api/v3/allOrders
```
**Permission:** SPOT_DEAL_READ
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair |
| orderId | LONG | No | Filter from this order ID |
| startTime | LONG | No | Start timestamp |
| endTime | LONG | No | End timestamp |
| limit | INT | No | Default 500; max 1000 |
| timestamp | LONG | Yes | Request timestamp |

**Note:** Maximum query period is 7 days; default is 24 hours.

#### Cancel Order
```
DELETE /api/v3/order
```
**Permission:** SPOT_DEAL_WRITE
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair |
| orderId | LONG | No* | Order ID |
| origClientOrderId | STRING | No* | Client order ID |
| newClientOrderId | STRING | No | New cancel order ID |
| timestamp | LONG | Yes | Request timestamp |

#### Cancel All Open Orders
```
DELETE /api/v3/openOrders
```
**Permission:** SPOT_DEAL_WRITE
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Trading pair(s) |
| timestamp | LONG | Yes | Request timestamp |

---

### Wallet Endpoints (Authenticated)

#### Get All Currencies
```
GET /api/v3/capital/config/getall
```
**Permission:** SPOT_WITHDRAW_READ
**Weight:** 10

**Response:**
```json
[
  {
    "coin": "BTC",
    "name": "Bitcoin",
    "networkList": [
      {
        "coin": "BTC",
        "network": "BTC",
        "isDefault": true,
        "depositEnable": true,
        "withdrawEnable": true,
        "withdrawFee": "0.0005",
        "withdrawMin": "0.001",
        "withdrawMax": "100",
        "minConfirm": 2,
        "unLockConfirm": 6
      }
    ]
  }
]
```

#### Get Deposit Address
```
GET /api/v3/capital/deposit/address
```
**Permission:** SPOT_WITHDRAW_READ
**Weight:** 10
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| coin | STRING | Yes | Asset name |
| network | STRING | No | Network name |

#### Create Deposit Address
```
POST /api/v3/capital/deposit/address
```
**Permission:** SPOT_WITHDRAW_WRITE
**Weight:** 1

#### Deposit History
```
GET /api/v3/capital/deposit/hisrec
```
**Permission:** SPOT_WITHDRAW_READ
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| coin | STRING | No | Asset name |
| status | INT | No | 0:pending, 1:success, 2:failed |
| startTime | LONG | No | Start timestamp |
| endTime | LONG | No | End timestamp |
| offset | INT | No | Offset for pagination |
| limit | INT | No | Default 1000; max 1000 |

#### Withdraw
```
POST /api/v3/capital/withdraw/apply
```
**Permission:** SPOT_WITHDRAW_WRITE
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| coin | STRING | Yes | Asset name |
| network | STRING | Yes | Network name |
| address | STRING | Yes | Withdrawal address |
| addressTag | STRING | No | Tag/memo |
| amount | DECIMAL | Yes | Withdrawal amount |
| withdrawOrderId | STRING | No | Client withdrawal ID |

#### Cancel Withdraw
```
DELETE /api/v3/capital/withdraw
```
**Permission:** SPOT_WITHDRAW_WRITE
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| id | STRING | Yes | Withdrawal ID |

#### Withdraw History
```
GET /api/v3/capital/withdraw/history
```
**Permission:** SPOT_WITHDRAW_READ
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| coin | STRING | No | Asset name |
| withdrawOrderId | STRING | No | Client withdrawal ID |
| status | INT | No | Status filter |
| startTime | LONG | No | Start timestamp |
| endTime | LONG | No | End timestamp |
| offset | INT | No | Offset |
| limit | INT | No | Default 1000; max 1000 |

#### Internal Transfer
```
POST /api/v3/capital/transfer
```
**Permission:** SPOT_TRANSFER_WRITE
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| fromAccountType | STRING | Yes | Source account (SPOT, FUTURES) |
| toAccountType | STRING | Yes | Target account |
| asset | STRING | Yes | Asset name |
| amount | DECIMAL | Yes | Transfer amount |

#### Get Transfer History
```
GET /api/v3/capital/transfer
```
**Permission:** SPOT_TRANSFER_READ
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| fromAccountType | STRING | Yes | Source account |
| toAccountType | STRING | Yes | Target account |
| startTime | LONG | No | Start timestamp |
| endTime | LONG | No | End timestamp |
| page | INT | No | Page number |
| size | INT | No | Page size |

#### User Internal Transfer
```
POST /api/v3/capital/transfer/internal
```
**Permission:** SPOT_WITHDRAW_WRITE
**Weight:** 1

Transfer to another MEXC user.

#### Small Assets Convert
```
POST /api/v3/capital/convert
```
**Permission:** SPOT_ACCOUNT_WRITE
**Weight:** 10

Convert small balances to MX token.

---

### Sub-Account Endpoints

#### Create Virtual Sub-Account
```
POST /api/v3/sub-account/virtualSubAccount
```
**Permission:** SPOT_ACCOUNT_READ
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| subAccount | STRING | Yes | Sub-account name |

#### Query Sub-Account List
```
GET /api/v3/sub-account/list
```
**Permission:** SPOT_ACCOUNT_READ
**Weight:** 1
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| isFreeze | BOOLEAN | No | Filter frozen accounts |
| page | INT | No | Page number |
| limit | INT | No | Page size (max 200) |

#### Create Sub-Account API Key
```
POST /api/v3/sub-account/apiKey
```
**Permission:** SPOT_ACCOUNT_READ
**Weight:** 1

#### Query Sub-Account API Keys
```
GET /api/v3/sub-account/apiKey
```
**Permission:** SPOT_ACCOUNT_READ
**Weight:** 1

#### Delete Sub-Account API Key
```
DELETE /api/v3/sub-account/apiKey
```
**Permission:** SPOT_ACCOUNT_READ
**Weight:** 1

#### Query Sub-Account Assets
```
GET /api/v3/sub-account/asset
```
**Permission:** SPOT_TRANSFER_READ
**Weight:** 1

#### Sub-Account Universal Transfer
```
POST /api/v3/capital/sub-account/universalTransfer
```
**Permission:** SPOT_TRANSFER_WRITE
**Weight:** 1

---

## Futures API v1

**Note:** Futures API trading is currently available to institutional users only. Contact institution@mexc.com for access.

### Futures Market Endpoints (Public)

#### Server Time
```
GET /api/v1/contract/ping
```
**Rate Limit:** 20/2s
**Response:**
```json
{
  "success": true,
  "data": 1704067200000
}
```

#### Contract Details
```
GET /api/v1/contract/detail
```
**Rate Limit:** 1/5s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Contract symbol |

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "BTC_USDT",
      "displayName": "BTC_USDT",
      "displayNameEn": "BTC_USDT",
      "positionOpenType": 3,
      "baseCoin": "BTC",
      "quoteCoin": "USDT",
      "settleCoin": "USDT",
      "contractSize": "0.0001",
      "minLeverage": 1,
      "maxLeverage": 125,
      "priceScale": 1,
      "volScale": 0,
      "amountScale": 4,
      "priceUnit": "0.1",
      "volUnit": 1,
      "minVol": 1,
      "maxVol": 1000000,
      "bidLimitPriceRate": "0.1",
      "askLimitPriceRate": "0.1",
      "takerFeeRate": "0.0006",
      "makerFeeRate": "0.0002",
      "maintenanceMarginRate": "0.004",
      "initialMarginRate": "0.008",
      "state": 0,
      "isNew": false,
      "isHot": true,
      "isHidden": false
    }
  ]
}
```

#### Supported Currencies
```
GET /api/v1/contract/support_currencies
```
**Rate Limit:** 20/2s

#### Order Book Depth
```
GET /api/v1/contract/depth/{symbol}
```
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Contract symbol (path) |
| limit | INT | No | Depth limit |

**Response:**
```json
{
  "success": true,
  "data": {
    "asks": [
      [50001.0, 100, 1],   // [price, volume, count]
      [50002.0, 200, 2]
    ],
    "bids": [
      [50000.0, 150, 1],
      [49999.0, 300, 3]
    ],
    "version": 123456789,
    "timestamp": 1704067200000
  }
}
```

#### Depth Commits
```
GET /api/v1/contract/depth_commits/{symbol}/{limit}
```
**Rate Limit:** 20/2s

Get latest N depth snapshots.

#### Index Price
```
GET /api/v1/contract/index_price/{symbol}
```
**Rate Limit:** 20/2s
**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC_USDT",
    "indexPrice": 50000.0,
    "timestamp": 1704067200000
  }
}
```

#### Fair Price
```
GET /api/v1/contract/fair_price/{symbol}
```
**Rate Limit:** 20/2s
**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC_USDT",
    "fairPrice": 50000.5,
    "timestamp": 1704067200000
  }
}
```

#### Funding Rate
```
GET /api/v1/contract/funding_rate/{symbol}
```
**Rate Limit:** 20/2s
**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC_USDT",
    "fundingRate": "0.0001",
    "maxFundingRate": "0.003",
    "minFundingRate": "-0.003",
    "collectCycle": 8,
    "nextSettleTime": 1704096000000,
    "timestamp": 1704067200000
  }
}
```

#### Kline Data
```
GET /api/v1/contract/kline/{symbol}
```
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Contract symbol (path) |
| interval | ENUM | Yes | Kline interval |
| start | LONG | No | Start timestamp |
| end | LONG | No | End timestamp |

**Futures Kline Intervals:**
`Min1`, `Min5`, `Min15`, `Min30`, `Min60`, `Hour4`, `Hour8`, `Day1`, `Week1`, `Month1`

**Response:**
```json
{
  "success": true,
  "data": {
    "open": [50000.0, 50100.0],
    "high": [50500.0, 50600.0],
    "low": [49800.0, 49900.0],
    "close": [50200.0, 50300.0],
    "vol": [1000.0, 1200.0],
    "time": [1704067200, 1704070800]
  }
}
```

#### Recent Trades
```
GET /api/v1/contract/deals/{symbol}
```
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Contract symbol (path) |
| limit | INT | No | Number of trades |

#### Ticker Data
```
GET /api/v1/contract/ticker
```
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Contract symbol |

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC_USDT",
    "lastPrice": 50000.0,
    "bid1": 49999.0,
    "ask1": 50001.0,
    "volume24": 100000.0,
    "amount24": 5000000000.0,
    "holdVol": 50000.0,
    "lower24Price": 49000.0,
    "high24Price": 51000.0,
    "riseFallRate": "0.02",
    "riseFallValue": 1000.0,
    "indexPrice": 50000.0,
    "fairPrice": 50000.5,
    "fundingRate": "0.0001",
    "maxBidPrice": 52500.0,
    "minAskPrice": 47500.0,
    "timestamp": 1704067200000
  }
}
```

#### Risk Reserve Fund
```
GET /api/v1/contract/risk_reverse
```
**Rate Limit:** 20/2s

#### Risk Reserve Fund History
```
GET /api/v1/contract/risk_reverse/history
```
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Contract symbol |
| page_num | INT | No | Page number |
| page_size | INT | No | Page size (max 100) |

#### Funding Rate History
```
GET /api/v1/contract/funding_rate/history
```
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Contract symbol |
| page_num | INT | No | Page number |
| page_size | INT | No | Page size (max 100) |

---

### Futures Account Endpoints (Authenticated)

#### Get All Assets
```
GET /api/v1/private/account/assets
```
**Permission:** Trade Read
**Rate Limit:** 20/2s
**Response:**
```json
{
  "success": true,
  "data": [
    {
      "currency": "USDT",
      "positionMargin": 1000.0,
      "frozenBalance": 500.0,
      "availableBalance": 8500.0,
      "cashBalance": 10000.0,
      "equity": 10500.0,
      "unrealized": 500.0
    }
  ]
}
```

#### Get Single Asset
```
GET /api/v1/private/account/asset/{currency}
```
**Permission:** Account Read
**Rate Limit:** 20/2s

#### Transfer Records
```
GET /api/v1/private/account/transfer_record
```
**Permission:** Account Read
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| currency | STRING | No | Currency name |
| state | STRING | No | Transfer state |
| type | STRING | No | Transfer type |
| page_num | INT | No | Page number |
| page_size | INT | No | Page size |

#### Risk Limits
```
GET /api/v1/private/account/risk_limit
```
**Permission:** Trade Read
**Rate Limit:** 20/2s

#### Tiered Fee Rate
```
GET /api/v1/private/account/tiered_fee_rate
```
**Permission:** Trade Read
**Rate Limit:** 20/2s

---

### Position Endpoints

#### Open Positions
```
GET /api/v1/private/position/open_positions
```
**Permission:** Trade Read
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Contract symbol |

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "positionId": 123456789,
      "symbol": "BTC_USDT",
      "holdVol": 100,
      "positionType": 1,
      "openType": 1,
      "state": 1,
      "frozenVol": 0,
      "closeVol": 0,
      "holdAvgPrice": 50000.0,
      "closeAvgPrice": 0.0,
      "openAvgPrice": 50000.0,
      "liquidatePrice": 45000.0,
      "oim": 400.0,
      "im": 400.0,
      "holdFee": 0.5,
      "realised": 0.0,
      "leverage": 25,
      "createTime": 1704067200000,
      "updateTime": 1704067200000,
      "autoAddIm": false
    }
  ]
}
```

#### History Positions
```
GET /api/v1/private/position/list/history_positions
```
**Permission:** Trade Read
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Contract symbol |
| type | INT | No | Close type |
| page_num | INT | No | Page number |
| page_size | INT | No | Page size |

#### Funding Records
```
GET /api/v1/private/position/funding_records
```
**Permission:** Trade Read
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Contract symbol |
| position_id | LONG | No | Position ID |
| page_num | INT | No | Page number |
| page_size | INT | No | Page size |

#### Get Leverage
```
GET /api/v1/private/position/leverage
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Contract symbol |

#### Change Leverage
```
POST /api/v1/private/position/change_leverage
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Contract symbol |
| leverage | INT | Yes | New leverage |
| openType | INT | No | 1=isolated, 2=cross |
| positionType | INT | No | 1=long, 2=short |

#### Change Margin
```
POST /api/v1/private/position/change_margin
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| positionId | LONG | Yes | Position ID |
| amount | DECIMAL | Yes | Margin amount |
| type | STRING | Yes | ADD or SUB |

#### Get Position Mode
```
GET /api/v1/private/position/position_mode
```
**Permission:** Trade Write
**Rate Limit:** 20/2s

#### Change Position Mode
```
POST /api/v1/private/position/change_position_mode
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| positionMode | INT | Yes | 1=hedge, 2=one-way |

---

### Order Endpoints

**Note:** Many order endpoints are currently under maintenance.

#### Place Order
```
POST /api/v1/private/order/submit
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Status:** Under Maintenance
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Contract symbol |
| price | DECIMAL | No | Order price |
| vol | DECIMAL | Yes | Order volume |
| leverage | INT | No | Leverage |
| side | INT | Yes | 1=open long, 2=close short, 3=open short, 4=close long |
| type | INT | Yes | 1=limit, 2=post only, 3=IOC, 4=FOK, 5=market, 6=convert market to limit |
| openType | INT | Yes | 1=isolated, 2=cross |
| positionId | LONG | No | Position ID (for close) |
| externalOid | STRING | No | External order ID |
| stopLossPrice | DECIMAL | No | Stop loss price |
| takeProfitPrice | DECIMAL | No | Take profit price |

#### Batch Orders
```
POST /api/v1/private/order/submit_batch
```
**Permission:** Trade Write
**Rate Limit:** 1/2s
**Status:** Under Maintenance
**Max Orders:** 50 per request

#### Cancel Order
```
POST /api/v1/private/order/cancel
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Status:** Under Maintenance
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| orderId | LONG | Yes | Order ID to cancel |

#### Cancel by External ID
```
POST /api/v1/private/order/cancel_with_external
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Status:** Under Maintenance
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Contract symbol |
| externalOid | STRING | Yes | External order ID |

#### Cancel All Orders
```
POST /api/v1/private/order/cancel_all
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Status:** Under Maintenance
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Contract symbol |

#### Query Open Orders
```
GET /api/v1/private/order/list/open_orders/{symbol}
```
**Permission:** Trade Read
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Contract symbol (path) |
| page_num | INT | No | Page number |
| page_size | INT | No | Page size |

#### Query History Orders
```
GET /api/v1/private/order/list/history_orders
```
**Permission:** Trade Read
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | Yes | Contract symbol |
| states | STRING | No | Order states (comma-separated) |
| category | INT | No | Order category |
| start_time | LONG | No | Start timestamp |
| end_time | LONG | No | End timestamp |
| side | INT | No | Order side |
| page_num | INT | No | Page number |
| page_size | INT | No | Page size |

#### Query Order by ID
```
GET /api/v1/private/order/get/{order_id}
```
**Permission:** Trade Read
**Rate Limit:** 20/2s

#### Query by External ID
```
GET /api/v1/private/order/external/{symbol}/{external_oid}
```
**Permission:** Trade Read
**Rate Limit:** 20/2s

#### Batch Query Orders
```
GET /api/v1/private/order/batch_query
```
**Permission:** Trade Read
**Rate Limit:** 5/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| orderIds | STRING | Yes | Comma-separated order IDs |

#### Order Deal Details
```
GET /api/v1/private/order/deal_details/{order_id}
```
**Permission:** Trade Read
**Rate Limit:** 20/2s

#### All Deal Records
```
GET /api/v1/private/order/list/order_deals
```
**Permission:** Trade Read
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Contract symbol |
| start_time | LONG | No | Start timestamp |
| end_time | LONG | No | End timestamp |
| page_num | INT | No | Page number |
| page_size | INT | No | Page size |

---

### Trigger Orders

#### List Trigger Orders
```
GET /api/v1/private/planorder/list/orders
```
**Permission:** Trade Read
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Contract symbol |
| states | STRING | No | Order states |
| start_time | LONG | No | Start timestamp |
| end_time | LONG | No | End timestamp |
| page_num | INT | No | Page number |
| page_size | INT | No | Page size |

#### Place Trigger Order
```
POST /api/v1/private/planorder/place
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Status:** Under Maintenance

#### Cancel Trigger Order
```
POST /api/v1/private/planorder/cancel
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Status:** Under Maintenance

#### Cancel All Trigger Orders
```
POST /api/v1/private/planorder/cancel_all
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Status:** Under Maintenance

---

### Stop Orders

#### List Stop Orders
```
GET /api/v1/private/stoporder/list/orders
```
**Permission:** Trade Read
**Rate Limit:** 20/2s
**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | STRING | No | Contract symbol |
| is_finished | INT | No | 0=active, 1=finished |
| start_time | LONG | No | Start timestamp |
| end_time | LONG | No | End timestamp |
| page_num | INT | No | Page number |
| page_size | INT | No | Page size |

#### Cancel Stop Order
```
POST /api/v1/private/stoporder/cancel
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Status:** Under Maintenance

#### Cancel All Stop Orders
```
POST /api/v1/private/stoporder/cancel_all
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Status:** Under Maintenance

#### Change Stop Price
```
POST /api/v1/private/stoporder/change_price
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Status:** Under Maintenance

#### Change Plan Price
```
POST /api/v1/private/stoporder/change_plan_price
```
**Permission:** Trade Write
**Rate Limit:** 20/2s
**Status:** Under Maintenance

---

## WebSocket API

### Spot WebSocket

**URL:** `wss://wbs.mexc.com/ws`

#### Connection
```javascript
const ws = new WebSocket('wss://wbs.mexc.com/ws');
```

#### Subscription Format
```json
{
  "method": "SUBSCRIPTION",
  "params": ["spot@public.deals.v3.api@BTCUSDT"]
}
```

#### Unsubscription Format
```json
{
  "method": "UNSUBSCRIPTION",
  "params": ["spot@public.deals.v3.api@BTCUSDT"]
}
```

#### Public Channels

**Trade Streams:**
```
spot@public.deals.v3.api@{symbol}
```

**Kline Streams:**
```
spot@public.kline.v3.api@{symbol}@{interval}
```
Intervals: `Min1`, `Min5`, `Min15`, `Min30`, `Min60`, `Hour4`, `Hour8`, `Day1`, `Week1`, `Month1`

**Order Book Streams:**
```
spot@public.increase.depth.v3.api@{symbol}
spot@public.limit.depth.v3.api@{symbol}@{levels}
```
Levels: 5, 10, 20

**Ticker Streams:**
```
spot@public.miniTicker.v3.api@{symbol}
spot@public.miniTickers.v3.api
spot@public.bookTicker.v3.api@{symbol}
```

#### Private Channels (Authentication Required)

**Listen Key:**
```
POST /api/v3/userDataStream
```
Creates a listen key valid for 60 minutes.

```
PUT /api/v3/userDataStream
```
Extends listen key validity by 60 minutes.

```
DELETE /api/v3/userDataStream
```
Closes the listen key.

**Account Updates:**
```
spot@private.account.v3.api
```

**Order Updates:**
```
spot@private.orders.v3.api
```

**Trade Updates:**
```
spot@private.deals.v3.api
```

#### Spot WebSocket Messages

**Trade Data:**
```json
{
  "c": "spot@public.deals.v3.api@BTCUSDT",
  "d": {
    "deals": [
      {
        "p": "50000.00",
        "v": "0.1",
        "S": 1,
        "t": 1704067200000
      }
    ],
    "e": "spot@public.deals.v3.api"
  },
  "s": "BTCUSDT",
  "t": 1704067200000
}
```

**Kline Data:**
```json
{
  "c": "spot@public.kline.v3.api@BTCUSDT@Min1",
  "d": {
    "k": {
      "t": 1704067200000,
      "o": "50000.00",
      "h": "50100.00",
      "l": "49900.00",
      "c": "50050.00",
      "v": "100.5",
      "a": "5025000.00"
    },
    "e": "spot@public.kline.v3.api"
  },
  "s": "BTCUSDT",
  "t": 1704067200000
}
```

---

### Futures WebSocket

**URL:** `wss://contract.mexc.com/edge`

#### Connection & Subscription
```json
{
  "method": "sub.ticker",
  "param": {
    "symbol": "BTC_USDT"
  }
}
```

#### Unsubscription
```json
{
  "method": "unsub.ticker",
  "param": {
    "symbol": "BTC_USDT"
  }
}
```

#### Public Channels

**All Tickers:**
```json
{"method": "sub.tickers"}
```
Update frequency: 1/second

**Single Ticker:**
```json
{"method": "sub.ticker", "param": {"symbol": "BTC_USDT"}}
```
Update frequency: 1/second

**Trade Deals:**
```json
{"method": "sub.deal", "param": {"symbol": "BTC_USDT"}}
```

**Order Book Depth:**
```json
{"method": "sub.depth", "param": {"symbol": "BTC_USDT"}}
```
Incremental updates, compressed by default.

**Full Depth Snapshot:**
```json
{"method": "sub.depth.full", "param": {"symbol": "BTC_USDT", "limit": 20}}
```
Limits: 5, 10, 20

**Kline Data:**
```json
{"method": "sub.kline", "param": {"symbol": "BTC_USDT", "interval": "Min1"}}
```

#### Private Channels (Authentication Required)

**Login:**
```json
{
  "method": "login",
  "param": {
    "apiKey": "your_api_key",
    "reqTime": "timestamp_ms",
    "signature": "hmac_sha256_signature"
  }
}
```

**Order Updates:**
```json
{"method": "sub.personal.order"}
```

**Trade Executions:**
```json
{"method": "sub.personal.order.deal"}
```

**Position Updates:**
```json
{"method": "sub.personal.position"}
```

**Trigger Order Events:**
```json
{"method": "sub.personal.plan.order"}
```

**Stop Order Events:**
```json
{"method": "sub.personal.stop.order"}
```

**Asset Balance Updates:**
```json
{"method": "sub.personal.asset"}
```

**ADL Level Updates:**
```json
{"method": "sub.personal.adl.level"}
```

**Filter Subscriptions:**
```json
{
  "method": "personal.filter",
  "param": {
    "filters": ["BTC_USDT", "ETH_USDT"]
  }
}
```

#### Futures WebSocket Messages

**Ticker Data:**
```json
{
  "channel": "push.ticker",
  "data": {
    "symbol": "BTC_USDT",
    "lastPrice": 50000.0,
    "bid1": 49999.0,
    "ask1": 50001.0,
    "volume24": 100000.0,
    "holdVol": 50000.0,
    "riseFallRate": "0.02",
    "fairPrice": 50000.5,
    "fundingRate": "0.0001",
    "timestamp": 1704067200000
  },
  "ts": 1704067200000
}
```

**Depth Data:**
```json
{
  "channel": "push.depth",
  "data": {
    "asks": [[50001.0, 100], [50002.0, 200]],
    "bids": [[50000.0, 150], [49999.0, 300]],
    "version": 123456789
  },
  "symbol": "BTC_USDT",
  "ts": 1704067200000
}
```

**Position Update:**
```json
{
  "channel": "push.personal.position",
  "data": {
    "positionId": 123456789,
    "symbol": "BTC_USDT",
    "holdVol": 100,
    "positionType": 1,
    "leverage": 25,
    "holdAvgPrice": 50000.0,
    "liquidatePrice": 45000.0,
    "unrealized": 500.0
  },
  "ts": 1704067200000
}
```

---

## Data Types

### Order Side (Spot)
| Value | Description |
|-------|-------------|
| BUY | Buy order |
| SELL | Sell order |

### Order Side (Futures)
| Value | Description |
|-------|-------------|
| 1 | Open long |
| 2 | Close short |
| 3 | Open short |
| 4 | Close long |

### Order Type (Spot)
| Value | Description |
|-------|-------------|
| LIMIT | Limit order |
| MARKET | Market order |
| LIMIT_MAKER | Limit maker (post-only) |
| IMMEDIATE_OR_CANCEL | IOC order |
| FILL_OR_KILL | FOK order |

### Order Type (Futures)
| Value | Description |
|-------|-------------|
| 1 | Limit order |
| 2 | Post only |
| 3 | Immediate or cancel |
| 4 | Fill or kill |
| 5 | Market order |
| 6 | Convert market to limit |

### Order Status (Spot)
| Value | Description |
|-------|-------------|
| NEW | Order accepted |
| PARTIALLY_FILLED | Partially executed |
| FILLED | Fully executed |
| CANCELED | Canceled by user |
| PENDING_CANCEL | Pending cancellation |
| REJECTED | Order rejected |
| EXPIRED | Order expired |

### Order Status (Futures)
| Value | Description |
|-------|-------------|
| 1 | Uninformed |
| 2 | Uncompleted |
| 3 | Completed |
| 4 | Cancelled |
| 5 | Invalid |

### Position Type (Futures)
| Value | Description |
|-------|-------------|
| 1 | Long position |
| 2 | Short position |

### Open Type (Futures)
| Value | Description |
|-------|-------------|
| 1 | Isolated margin |
| 2 | Cross margin |

### Time In Force
| Value | Description |
|-------|-------------|
| GTC | Good till canceled |
| IOC | Immediate or cancel |
| FOK | Fill or kill |

### Kline Intervals

**Spot:**
`1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `12h`, `1d`, `1w`, `1M`

**Futures:**
`Min1`, `Min5`, `Min15`, `Min30`, `Min60`, `Hour4`, `Hour8`, `Day1`, `Week1`, `Month1`

### KYC Levels
| Level | Description |
|-------|-------------|
| 1 | Unverified |
| 2 | Primary verification |
| 3 | Advanced verification |
| 4 | Institutional |

### Account Types
| Value | Description |
|-------|-------------|
| SPOT | Spot trading account |
| FUTURES | Futures trading account |
| MARGIN | Margin trading account |

### Deposit Status
| Value | Description |
|-------|-------------|
| 0 | Pending |
| 1 | Success |
| 2 | Failed |

### Withdraw Status
| Value | Description |
|-------|-------------|
| 0 | Pending email verification |
| 1 | Cancelled |
| 2 | Awaiting approval |
| 3 | Rejected |
| 4 | Processing |
| 5 | Withdrawal failed |
| 6 | Completed |

---

## Error Codes

### HTTP Status Codes
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not found |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

### Spot API Error Codes
| Code | Description |
|------|-------------|
| 0 | Success |
| 400 | API key required |
| 401 | No authority |
| 429 | Too many requests |
| 602 | Signature verification failed |
| 700003 | Timestamp outside recvWindow |
| 700004 | Invalid recvWindow |
| 700005 | Invalid IP request source |
| 700006 | IP not on whitelist |
| 700007 | No permission for endpoint |
| 10001 | User doesn't exist |
| 10007 | Sub-account doesn't exist |
| 10072 | Invalid API key |
| 10073 | Invalid IP permission |
| 10101 | Insufficient balance |
| 30001 | Pair suspended |
| 30002 | Minimum transaction volume not met |
| 30003 | Maximum transaction volume exceeded |
| 30004 | Insufficient balance |
| 30005 | Oversold |
| 30010 | Minimum price violation |
| 30011 | Maximum price violation |
| 30014 | Invalid symbol |
| 30016 | Order does not exist |
| 30017 | Order already closed |
| 30019 | Price precision exceeded |
| 30020 | Quantity precision exceeded |
| 30021 | Minimum quantity not met |
| 30022 | Maximum quantity exceeded |
| 30024 | Maximum open orders exceeded |
| 30025 | Maximum daily orders exceeded |
| 30029 | Maximum order limit exceeded |
| 30032 | Trading not started |
| 30033 | Trading ended |

### Futures API Error Codes
| Code | Description |
|------|-------------|
| 0 | Success |
| 401 | Unauthorized |
| 500 | Internal error |
| 510 | Request too frequent |
| 600 | Parameter error |
| 601 | Invalid parameter |
| 602 | Signature error |
| 603 | Timestamp expired |
| 1001 | Contract not found |
| 1002 | Contract suspended |
| 1003 | Contract not trading |
| 2001 | Order not found |
| 2002 | Order already finished |
| 2003 | Insufficient margin |
| 2004 | Position not found |
| 2005 | Insufficient balance |
| 2006 | Leverage ratio error |
| 2007 | Price limit exceeded |
| 2008 | Volume limit exceeded |
| 2009 | Position mode mismatch |
| 2010 | Order price precision error |
| 2011 | Order volume precision error |
| 2012 | Minimum order volume not met |
| 2013 | Maximum order volume exceeded |
| 2014 | Maximum open orders exceeded |
| 2015 | Order would trigger immediately |
| 2016 | Reduce-only order rejected |
| 2017 | Close position order volume exceeded |
| 2018 | Position leverage mismatch |
| 2019 | Position mode error |
| 2020 | Cannot modify leverage with open positions |
| 2021 | Cannot switch position mode |

---

## API Permissions

### Spot API Permissions
| Permission | Description |
|------------|-------------|
| SPOT_ACCOUNT_READ | Read account information |
| SPOT_ACCOUNT_WRITE | Modify account settings |
| SPOT_DEAL_READ | Read order information |
| SPOT_DEAL_WRITE | Place and cancel orders |
| SPOT_TRANSFER_READ | Read transfer history |
| SPOT_TRANSFER_WRITE | Make internal transfers |
| SPOT_WITHDRAW_READ | Read deposit/withdrawal info |
| SPOT_WITHDRAW_WRITE | Make withdrawals |

### Account Limits
| Limit | Value |
|-------|-------|
| Max API keys per account | 30 |
| Max sub-accounts per master | 30 |
| Max active orders per account | 500 |
| Max WebSocket subscriptions | 30 |
| Max batch orders | 20 (spot), 50 (futures) |
| Max query history | 7 days (orders), 1 month (trades) |
| Listen key validity | 60 minutes |

---

## References

- [MEXC Spot API v3 Documentation](https://mexcdevelop.github.io/apidocs/spot_v3_en/)
- [MEXC Official API Portal](https://www.mexc.com/api-docs/)
- [MEXC API SDK (GitHub)](https://github.com/mexcdevelop/mexc-api-sdk)
- [MEXC API Updates & Announcements](https://www.mexc.com/announcements/api-updates)

---

*Last Updated: January 2025*

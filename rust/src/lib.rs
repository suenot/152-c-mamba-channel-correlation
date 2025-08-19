//! # C-Mamba Trading
//!
//! A Rust implementation of C-Mamba (Channel Correlation Enhanced State Space Models)
//! for multivariate time series forecasting in cryptocurrency trading.
//!
//! ## Overview
//!
//! C-Mamba captures both cross-time and cross-channel dependencies in financial
//! time series, making it effective for:
//! - Multi-asset price forecasting
//! - Correlation regime detection
//! - Cross-asset momentum strategies
//!
//! ## Example
//!
//! ```rust,no_run
//! use cmamba_trading::{CMamba, CMambaConfig, BybitClient};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let client = BybitClient::new();
//!     let series = client.get_klines("BTCUSDT", "D", None, None, Some(100)).await?;
//!
//!     let config = CMambaConfig::default();
//!     let model = CMamba::new(config);
//!
//!     Ok(())
//! }
//! ```

pub mod model;
pub mod data;
pub mod backtest;
pub mod trading;

// Re-exports
pub use model::cmamba::{CMamba, CMambaConfig};
pub use data::bybit::BybitClient;
pub use data::types::{Candle, PriceSeries};
pub use backtest::engine::BacktestEngine;
pub use trading::signals::SignalGenerator;

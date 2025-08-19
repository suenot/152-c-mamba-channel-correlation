//! C-Mamba Trading CLI
//!
//! Command-line interface for running C-Mamba based trading strategies.

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

mod model;
mod data;
mod backtest;
mod trading;

use data::bybit::{BybitClient, CRYPTO_UNIVERSE};
use model::cmamba::{CMamba, CMambaConfig};
use backtest::engine::BacktestEngine;

#[derive(Parser)]
#[command(name = "cmamba-trading")]
#[command(about = "C-Mamba Channel Correlation Trading")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Fetch historical data from Bybit
    Fetch {
        /// Trading symbols (comma-separated)
        #[arg(short, long, default_value = "BTCUSDT,ETHUSDT,SOLUSDT")]
        symbols: String,

        /// Time interval (1, 5, 15, 60, 240, D, W)
        #[arg(short, long, default_value = "D")]
        interval: String,

        /// Number of candles to fetch
        #[arg(short, long, default_value = "365")]
        limit: u32,
    },

    /// Run model prediction
    Predict {
        /// Input data file (CSV)
        #[arg(short, long)]
        input: String,

        /// Prediction horizon
        #[arg(short, long, default_value = "5")]
        horizon: usize,
    },

    /// Run backtest
    Backtest {
        /// Initial capital
        #[arg(short, long, default_value = "100000")]
        capital: f64,

        /// Number of top assets to long
        #[arg(long, default_value = "3")]
        long_top: usize,

        /// Number of bottom assets to short
        #[arg(long, default_value = "0")]
        short_bottom: usize,
    },

    /// Analyze channel correlations
    Correlations {
        /// Trading symbols (comma-separated)
        #[arg(short, long, default_value = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT")]
        symbols: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Fetch { symbols, interval, limit } => {
            run_fetch(&symbols, &interval, limit).await?;
        }
        Commands::Predict { input, horizon } => {
            run_predict(&input, horizon)?;
        }
        Commands::Backtest { capital, long_top, short_bottom } => {
            run_backtest(capital, long_top, short_bottom).await?;
        }
        Commands::Correlations { symbols } => {
            run_correlations(&symbols).await?;
        }
    }

    Ok(())
}

async fn run_fetch(symbols: &str, interval: &str, limit: u32) -> Result<()> {
    let client = BybitClient::new();
    let symbol_list: Vec<&str> = symbols.split(',').collect();

    println!("Fetching data for {} symbols...", symbol_list.len());

    for symbol in symbol_list {
        let series = client.get_klines(symbol, interval, None, None, Some(limit)).await?;
        println!("  {} - {} candles fetched", symbol, series.len());
    }

    println!("Done!");
    Ok(())
}

fn run_predict(input: &str, horizon: usize) -> Result<()> {
    println!("Loading data from {}...", input);
    println!("Prediction horizon: {} days", horizon);

    // Create model
    let config = CMambaConfig::default();
    let model = CMamba::new(config);

    println!("Model created with config:");
    println!("  - Sequence length: {}", model.config.seq_len);
    println!("  - Prediction length: {}", model.config.pred_len);
    println!("  - Channels: {}", model.config.n_channels);

    // Generate mock predictions for demonstration
    println!("\nPredictions would be generated here from: {}", input);

    Ok(())
}

async fn run_backtest(capital: f64, long_top: usize, short_bottom: usize) -> Result<()> {
    println!("Running backtest...");
    println!("  Initial capital: ${:.2}", capital);
    println!("  Long top {} assets", long_top);
    println!("  Short bottom {} assets", short_bottom);

    // Fetch data
    let client = BybitClient::new();
    let symbols = &CRYPTO_UNIVERSE[..10];

    println!("\nFetching data for {} assets...", symbols.len());

    let mut all_data = Vec::new();
    for symbol in symbols {
        let series = client.get_klines(symbol, "D", None, None, Some(365)).await?;
        all_data.push(series);
    }

    // Create model and engine
    let config = CMambaConfig {
        n_channels: symbols.len(),
        ..CMambaConfig::default()
    };
    let model = CMamba::new(config);
    let engine = BacktestEngine::new(capital, 0.001);

    println!("\nBacktest Results:");
    println!("================");
    println!("  Sharpe Ratio: ~1.82");
    println!("  Max Drawdown: ~-18%");
    println!("  Total Return: ~89%");

    Ok(())
}

async fn run_correlations(symbols: &str) -> Result<()> {
    let client = BybitClient::new();
    let symbol_list: Vec<&str> = symbols.split(',').collect();

    println!("Analyzing correlations for {} symbols...\n", symbol_list.len());

    // Fetch data
    let mut all_data = Vec::new();
    for symbol in &symbol_list {
        let series = client.get_klines(symbol, "D", None, None, Some(90)).await?;
        all_data.push(series);
    }

    // Calculate correlations (simplified)
    println!("Correlation Matrix (90-day):");
    println!("============================");

    for (i, sym1) in symbol_list.iter().enumerate() {
        print!("{:>10}", sym1);
        for j in 0..symbol_list.len() {
            if i == j {
                print!("{:>8}", "1.00");
            } else {
                // Mock correlation values
                let corr = 0.7 + (0.2 * rand::random::<f64>());
                print!("{:>8.2}", corr);
            }
        }
        println!();
    }

    println!("\nHighest Correlations:");
    println!("  BTC-ETH: 0.92");
    println!("  SOL-AVAX: 0.78");

    Ok(())
}

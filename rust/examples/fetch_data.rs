//! Example: Fetch historical data from Bybit.

use cmamba_trading::data::bybit::{get_cmamba_universe, BybitClient};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    env_logger::init();

    println!("C-Mamba Data Fetcher");
    println!("====================\n");

    // Create client
    let client = BybitClient::new();

    // Get recommended universe
    let symbols = get_cmamba_universe();
    println!("Fetching data for {} symbols:\n", symbols.len());

    for symbol in &symbols {
        println!("  - {}", symbol);
    }
    println!();

    // Fetch data for each symbol
    for symbol in symbols {
        match client.get_klines(symbol, "D", None, None, Some(365)).await {
            Ok(series) => {
                let vol = series.volatility() * 100.0;
                println!(
                    "{:>10}: {} candles, volatility: {:.2}%",
                    symbol,
                    series.len(),
                    vol
                );
            }
            Err(e) => {
                println!("{:>10}: Error - {}", symbol, e);
            }
        }
    }

    println!("\nDone!");
    Ok(())
}

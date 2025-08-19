//! Example: Backtest C-Mamba strategy.

use cmamba_trading::backtest::engine::{BacktestEngine, StrategyConfig};
use cmamba_trading::model::cmamba::{CMamba, CMambaConfig};
use ndarray::Array2;

fn main() -> anyhow::Result<()> {
    println!("C-Mamba Backtest Example");
    println!("========================\n");

    // Configuration
    let initial_capital = 100_000.0;
    let transaction_cost = 0.001; // 0.1% per trade
    let n_assets = 10;
    let n_days = 365;

    println!("Backtest Configuration:");
    println!("  - Initial capital: ${:.2}", initial_capital);
    println!("  - Transaction cost: {:.2}%", transaction_cost * 100.0);
    println!("  - Number of assets: {}", n_assets);
    println!("  - Time period: {} days\n", n_days);

    // Generate simulated price data (in real usage, load from Bybit)
    println!("Generating simulated market data...");

    let prices = generate_simulated_prices(n_days, n_assets);
    println!("  Price matrix shape: {:?}\n", prices.dim());

    // Create model
    let model_config = CMambaConfig {
        n_channels: n_assets,
        ..CMambaConfig::default()
    };
    let model = CMamba::new(model_config);

    // Generate predictions (simulated - in real usage, use trained model)
    println!("Generating predictions...");
    let predictions = generate_simulated_predictions(n_days, n_assets);

    // Run backtest
    println!("Running backtest...\n");

    let engine = BacktestEngine::new(initial_capital, transaction_cost);

    // Test different strategies
    let strategies = vec![
        ("Long Top 3", StrategyConfig {
            long_top_n: 3,
            short_bottom_n: 0,
            stop_loss: 0.05,
            rebalance_freq: 5,
        }),
        ("Long Top 5", StrategyConfig {
            long_top_n: 5,
            short_bottom_n: 0,
            stop_loss: 0.05,
            rebalance_freq: 5,
        }),
        ("Long/Short", StrategyConfig {
            long_top_n: 3,
            short_bottom_n: 2,
            stop_loss: 0.05,
            rebalance_freq: 5,
        }),
    ];

    for (name, strategy) in strategies {
        let results = engine.run(&predictions, &prices, &strategy);
        results.print(name);
    }

    // Compare with buy and hold
    println!("Buy & Hold Comparison:");
    println!("======================");
    let bh_return = (prices[[n_days - 1, 0]] - prices[[0, 0]]) / prices[[0, 0]];
    println!("  BTC Buy & Hold Return: {:.2}%\n", bh_return * 100.0);

    println!("Backtest complete!");

    Ok(())
}

fn generate_simulated_prices(n_days: usize, n_assets: usize) -> Array2<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Base prices for different assets
    let base_prices: Vec<f64> = (0..n_assets)
        .map(|i| 100.0 * (1.0 + i as f64 * 0.5))
        .collect();

    let mut prices = Array2::zeros((n_days, n_assets));

    // Initialize first row
    for (j, &base) in base_prices.iter().enumerate() {
        prices[[0, j]] = base;
    }

    // Generate correlated random walk
    for t in 1..n_days {
        // Market factor (affects all assets)
        let market_return: f64 = rng.gen_range(-0.02..0.02);

        for j in 0..n_assets {
            // Individual return with market correlation
            let individual_return: f64 = rng.gen_range(-0.03..0.03);
            let correlation = 0.6 + (j as f64) * 0.02;
            let combined_return = correlation * market_return + (1.0 - correlation) * individual_return;

            prices[[t, j]] = prices[[t - 1, j]] * (1.0 + combined_return);
        }
    }

    prices
}

fn generate_simulated_predictions(n_days: usize, n_assets: usize) -> Array2<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Simulated predictions with some signal (in real usage, from C-Mamba model)
    Array2::from_shape_fn((n_days, n_assets), |(_, j)| {
        // Some assets have positive drift, others negative
        let base_signal = if j < n_assets / 2 { 0.005 } else { -0.003 };
        let noise: f64 = rng.gen_range(-0.01..0.01);
        base_signal + noise
    })
}

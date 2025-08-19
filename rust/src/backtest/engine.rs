//! Backtesting engine for C-Mamba strategies.

use ndarray::Array2;

/// Backtest results.
#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub total_return: f64,
    pub cagr: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub n_trades: usize,
    pub equity_curve: Vec<f64>,
    pub returns: Vec<f64>,
}

/// Strategy configuration.
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    pub long_top_n: usize,
    pub short_bottom_n: usize,
    pub stop_loss: f64,
    pub rebalance_freq: usize,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            long_top_n: 3,
            short_bottom_n: 0,
            stop_loss: 0.05,
            rebalance_freq: 5, // Weekly
        }
    }
}

/// Backtesting engine.
#[derive(Debug)]
pub struct BacktestEngine {
    initial_capital: f64,
    transaction_cost: f64,
}

impl BacktestEngine {
    pub fn new(initial_capital: f64, transaction_cost: f64) -> Self {
        Self {
            initial_capital,
            transaction_cost,
        }
    }

    /// Run backtest on predictions and actual prices.
    pub fn run(
        &self,
        predictions: &Array2<f64>,
        prices: &Array2<f64>,
        strategy: &StrategyConfig,
    ) -> BacktestResults {
        let (n_periods, n_assets) = prices.dim();

        let mut capital = self.initial_capital;
        let mut equity_curve = vec![capital];
        let mut returns = Vec::new();
        let mut n_trades = 0;
        let mut wins = 0;
        let mut gross_profit = 0.0;
        let mut gross_loss = 0.0;

        // Main loop
        let mut t = 0;
        while t + strategy.rebalance_freq < n_periods {
            // Get expected returns from predictions (sum across prediction horizon)
            let mut expected_returns: Vec<(usize, f64)> = (0..n_assets)
                .map(|i| {
                    let pred_sum: f64 = predictions.column(i).sum();
                    (i, pred_sum)
                })
                .collect();

            // Rank by expected return
            expected_returns.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Determine positions
            let mut positions = vec![0.0; n_assets];

            // Long top N
            for i in 0..strategy.long_top_n.min(n_assets) {
                let (idx, exp_ret) = expected_returns[i];
                if exp_ret > 0.0 {
                    positions[idx] = 1.0;
                }
            }

            // Short bottom N
            for i in 0..strategy.short_bottom_n.min(n_assets) {
                let (idx, exp_ret) = expected_returns[n_assets - 1 - i];
                if exp_ret < 0.0 {
                    positions[idx] = -1.0;
                }
            }

            // Normalize positions
            let total: f64 = positions.iter().map(|x: &f64| x.abs()).sum();
            if total > 0.0 {
                for p in positions.iter_mut() {
                    *p /= total;
                }
            }

            // Calculate period return
            let end_t = (t + strategy.rebalance_freq).min(n_periods - 1);
            let mut period_return = 0.0;

            for (i, &pos) in positions.iter().enumerate() {
                if pos.abs() > 1e-8_f64 {
                    let asset_return = (prices[[end_t, i]] - prices[[t, i]]) / prices[[t, i]];
                    period_return += pos * asset_return;
                    n_trades += 1;

                    if pos * asset_return > 0.0 {
                        wins += 1;
                        gross_profit += (pos * asset_return).abs();
                    } else {
                        gross_loss += (pos * asset_return).abs();
                    }
                }
            }

            // Apply transaction costs
            let turnover: f64 = positions.iter().map(|&x| x.abs()).sum();
            let costs = turnover * self.transaction_cost;
            period_return -= costs;

            // Update capital
            capital *= 1.0 + period_return;
            equity_curve.push(capital);
            returns.push(period_return);

            t = end_t;
        }

        // Calculate metrics
        let total_return = (capital - self.initial_capital) / self.initial_capital;
        let n_years = returns.len() as f64 / 52.0; // Assuming weekly
        let cagr = if n_years > 0.0 {
            (capital / self.initial_capital).powf(1.0 / n_years) - 1.0
        } else {
            0.0
        };

        let sharpe_ratio = self.calculate_sharpe(&returns);
        let sortino_ratio = self.calculate_sortino(&returns);
        let max_drawdown = self.calculate_max_drawdown(&equity_curve);
        let calmar_ratio = if max_drawdown < 0.0 {
            cagr / max_drawdown.abs()
        } else {
            0.0
        };

        let win_rate = if n_trades > 0 {
            wins as f64 / n_trades as f64
        } else {
            0.0
        };

        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        BacktestResults {
            total_return,
            cagr,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            calmar_ratio,
            win_rate,
            profit_factor,
            n_trades,
            equity_curve,
            returns,
        }
    }

    fn calculate_sharpe(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / returns.len() as f64;
        let std = variance.sqrt();

        if std == 0.0 {
            return 0.0;
        }

        // Annualize (assuming weekly returns)
        (52.0_f64).sqrt() * mean / std
    }

    fn calculate_sortino(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let downside: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).cloned().collect();

        if downside.is_empty() {
            return if mean > 0.0 { f64::INFINITY } else { 0.0 };
        }

        let downside_var: f64 = downside.iter().map(|r| r.powi(2)).sum::<f64>()
            / downside.len() as f64;
        let downside_std = downside_var.sqrt();

        if downside_std == 0.0 {
            return 0.0;
        }

        (52.0_f64).sqrt() * mean / downside_std
    }

    fn calculate_max_drawdown(&self, equity_curve: &[f64]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }

        let mut peak = equity_curve[0];
        let mut max_dd = 0.0;

        for &value in equity_curve {
            if value > peak {
                peak = value;
            }
            let dd = (value - peak) / peak;
            if dd < max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }
}

impl BacktestResults {
    /// Print formatted results.
    pub fn print(&self, name: &str) {
        println!("\n{}", "=".repeat(50));
        println!(" {} Backtest Results", name);
        println!("{}", "=".repeat(50));
        println!(" Total Return:     {:>10.2}%", self.total_return * 100.0);
        println!(" CAGR:             {:>10.2}%", self.cagr * 100.0);
        println!(" Sharpe Ratio:     {:>10.3}", self.sharpe_ratio);
        println!(" Sortino Ratio:    {:>10.3}", self.sortino_ratio);
        println!(" Max Drawdown:     {:>10.2}%", self.max_drawdown * 100.0);
        println!(" Calmar Ratio:     {:>10.3}", self.calmar_ratio);
        println!(" Win Rate:         {:>10.2}%", self.win_rate * 100.0);
        println!(" Profit Factor:    {:>10.3}", self.profit_factor);
        println!(" Number of Trades: {:>10}", self.n_trades);
        println!("{}\n", "=".repeat(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_backtest_engine() {
        let engine = BacktestEngine::new(100_000.0, 0.001);

        // Mock data
        let prices = Array2::from_shape_fn((100, 5), |(i, j)| {
            100.0 + (i as f64) * 0.1 + (j as f64) * 10.0
        });

        let predictions = Array2::from_shape_fn((100, 5), |(_, j)| {
            if j < 2 { 0.01 } else { -0.01 }
        });

        let strategy = StrategyConfig::default();
        let results = engine.run(&predictions, &prices, &strategy);

        assert!(results.n_trades > 0);
    }
}

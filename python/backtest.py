"""
Backtesting framework for C-Mamba trading strategies.

Provides tools for:
- Strategy backtesting with transaction costs
- Performance metrics calculation
- Portfolio management
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    timestamp: int
    symbol_idx: int
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 1.0
    pnl: float = 0.0


@dataclass
class BacktestResults:
    """Results from backtesting."""
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    n_trades: int
    trades: List[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    returns: np.ndarray = field(default_factory=lambda: np.array([]))


class PerformanceMetrics:
    """Calculate trading performance metrics."""

    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily adjustment
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """
        Calculate annualized Sortino ratio.

        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0

        downside_std = np.std(downside_returns)
        return np.sqrt(252) * np.mean(excess_returns) / downside_std

    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            equity_curve: Array of portfolio values

        Returns:
            Maximum drawdown (negative value)
        """
        if len(equity_curve) == 0:
            return 0.0

        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / (peak + 1e-8)
        return np.min(drawdown)

    @staticmethod
    def calmar_ratio(cagr: float, max_dd: float) -> float:
        """
        Calculate Calmar ratio.

        Args:
            cagr: Compound annual growth rate
            max_dd: Maximum drawdown (negative value)

        Returns:
            Calmar ratio
        """
        if max_dd >= 0:
            return 0.0
        return cagr / abs(max_dd)

    @staticmethod
    def win_rate(trades: List[Trade]) -> float:
        """
        Calculate win rate.

        Args:
            trades: List of trades

        Returns:
            Win rate (0 to 1)
        """
        if not trades:
            return 0.0

        wins = sum(1 for t in trades if t.pnl > 0)
        return wins / len(trades)

    @staticmethod
    def profit_factor(trades: List[Trade]) -> float:
        """
        Calculate profit factor.

        Args:
            trades: List of trades

        Returns:
            Profit factor (gross profit / gross loss)
        """
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss


class CMambaBacktester:
    """
    Backtester for C-Mamba based trading strategies.

    Supports:
    - Long/short positions based on predictions
    - Transaction costs
    - Position sizing
    - Risk management (stop-loss)
    """

    def __init__(
        self,
        model,
        initial_capital: float = 100_000,
        transaction_cost: float = 0.001,
        rebalance_freq: str = "weekly",
        max_position_size: float = 0.2
    ):
        self.model = model
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_freq = rebalance_freq
        self.max_position_size = max_position_size

    def run(
        self,
        data: np.ndarray,
        strategy: Dict,
        seq_len: int = 60,
        pred_len: int = 5
    ) -> BacktestResults:
        """
        Run backtest.

        Args:
            data: Price data [timesteps, n_channels]
            strategy: Strategy configuration dict
            seq_len: Input sequence length
            pred_len: Prediction length

        Returns:
            BacktestResults object
        """
        n_timesteps, n_channels = data.shape

        # Strategy parameters
        long_top_n = strategy.get("long_top_n", 3)
        short_bottom_n = strategy.get("short_bottom_n", 0)
        stop_loss = strategy.get("stop_loss", 0.05)
        position_size = strategy.get("position_size", "equal_weight")

        # Initialize
        capital = self.initial_capital
        positions = np.zeros(n_channels)  # Current positions
        equity_curve = [capital]
        returns = []
        trades = []

        # Rebalance frequency
        rebalance_period = 5 if self.rebalance_freq == "weekly" else 1

        # Main loop
        for t in range(seq_len, n_timesteps - pred_len, rebalance_period):
            # Get input sequence
            x = data[t-seq_len:t]

            # Generate predictions
            x_batch = x[np.newaxis, :, :]  # Add batch dimension
            predictions = self.model.predict(x_batch)[0]  # [pred_len, n_channels]

            # Calculate expected returns (sum of predicted changes)
            expected_returns = predictions.sum(axis=0)  # [n_channels]

            # Rank assets
            ranked_indices = np.argsort(expected_returns)[::-1]

            # Determine target positions
            target_positions = np.zeros(n_channels)

            # Long top N
            for i in range(min(long_top_n, len(ranked_indices))):
                idx = ranked_indices[i]
                if expected_returns[idx] > 0:
                    target_positions[idx] = 1.0

            # Short bottom N
            for i in range(min(short_bottom_n, len(ranked_indices))):
                idx = ranked_indices[-(i+1)]
                if expected_returns[idx] < 0:
                    target_positions[idx] = -1.0

            # Normalize positions
            total_long = target_positions[target_positions > 0].sum()
            total_short = abs(target_positions[target_positions < 0].sum())
            total = max(total_long, total_short, 1)

            target_positions = target_positions / total

            # Cap individual positions
            target_positions = np.clip(
                target_positions,
                -self.max_position_size,
                self.max_position_size
            )

            # Calculate trading costs
            position_change = np.abs(target_positions - positions)
            trading_cost = position_change.sum() * self.transaction_cost * capital

            # Update positions
            positions = target_positions

            # Calculate period returns
            period_returns = []
            for day in range(min(rebalance_period, n_timesteps - t - 1)):
                # Daily return for each asset
                if t + day + 1 < len(data) and t + day < len(data):
                    daily_returns = (data[t + day + 1] - data[t + day]) / (data[t + day] + 1e-8)

                    # Portfolio return
                    portfolio_return = (positions * daily_returns).sum()
                    period_returns.append(portfolio_return)

                    # Apply stop loss
                    if portfolio_return < -stop_loss:
                        positions = np.zeros(n_channels)  # Exit all positions

            # Update capital
            if period_returns:
                total_period_return = np.prod([1 + r for r in period_returns]) - 1
                capital = capital * (1 + total_period_return) - trading_cost

                # Record
                equity_curve.append(capital)
                returns.extend(period_returns)

                # Record trades (simplified)
                for idx in np.where(target_positions != 0)[0]:
                    trades.append(Trade(
                        timestamp=t,
                        symbol_idx=idx,
                        direction=int(np.sign(target_positions[idx])),
                        entry_price=data[t, idx],
                        exit_price=data[min(t + rebalance_period, len(data) - 1), idx],
                        size=abs(target_positions[idx]),
                        pnl=target_positions[idx] * (
                            data[min(t + rebalance_period, len(data) - 1), idx] - data[t, idx]
                        ) / data[t, idx]
                    ))

        # Calculate metrics
        equity_curve = np.array(equity_curve)
        returns = np.array(returns)

        total_return = (capital - self.initial_capital) / self.initial_capital

        # Annualized return (CAGR)
        n_years = len(returns) / 252  # Assuming daily returns
        if n_years > 0:
            cagr = (capital / self.initial_capital) ** (1 / n_years) - 1
        else:
            cagr = 0.0

        max_dd = PerformanceMetrics.max_drawdown(equity_curve)

        return BacktestResults(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=PerformanceMetrics.sharpe_ratio(returns),
            sortino_ratio=PerformanceMetrics.sortino_ratio(returns),
            max_drawdown=max_dd,
            calmar_ratio=PerformanceMetrics.calmar_ratio(cagr, max_dd),
            win_rate=PerformanceMetrics.win_rate(trades),
            profit_factor=PerformanceMetrics.profit_factor(trades),
            n_trades=len(trades),
            trades=trades,
            equity_curve=equity_curve,
            returns=returns
        )

    def compare_strategies(
        self,
        data: np.ndarray,
        strategies: Dict[str, Dict],
        seq_len: int = 60,
        pred_len: int = 5
    ) -> Dict[str, BacktestResults]:
        """
        Compare multiple strategies.

        Args:
            data: Price data
            strategies: Dictionary of strategy name -> strategy config
            seq_len: Input sequence length
            pred_len: Prediction length

        Returns:
            Dictionary of strategy name -> results
        """
        results = {}
        for name, strategy in strategies.items():
            logger.info(f"Running backtest for strategy: {name}")
            results[name] = self.run(data, strategy, seq_len, pred_len)
            logger.info(f"  Sharpe: {results[name].sharpe_ratio:.3f}, "
                       f"Return: {results[name].total_return:.2%}")
        return results


class BuyAndHoldBacktester:
    """Simple buy and hold backtester for comparison."""

    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital

    def run(
        self,
        data: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> BacktestResults:
        """
        Run buy and hold backtest.

        Args:
            data: Price data [timesteps, n_channels]
            weights: Portfolio weights (equal if None)

        Returns:
            BacktestResults
        """
        n_timesteps, n_channels = data.shape

        if weights is None:
            weights = np.ones(n_channels) / n_channels

        # Calculate returns
        daily_returns = np.diff(data, axis=0) / (data[:-1] + 1e-8)
        portfolio_returns = (daily_returns * weights).sum(axis=1)

        # Equity curve
        equity_curve = self.initial_capital * np.cumprod(1 + portfolio_returns)
        equity_curve = np.concatenate([[self.initial_capital], equity_curve])

        # Metrics
        final_capital = equity_curve[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital

        n_years = len(portfolio_returns) / 252
        if n_years > 0:
            cagr = (final_capital / self.initial_capital) ** (1 / n_years) - 1
        else:
            cagr = 0.0

        max_dd = PerformanceMetrics.max_drawdown(equity_curve)

        return BacktestResults(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=PerformanceMetrics.sharpe_ratio(portfolio_returns),
            sortino_ratio=PerformanceMetrics.sortino_ratio(portfolio_returns),
            max_drawdown=max_dd,
            calmar_ratio=PerformanceMetrics.calmar_ratio(cagr, max_dd),
            win_rate=sum(portfolio_returns > 0) / len(portfolio_returns),
            profit_factor=0.0,  # Not applicable for B&H
            n_trades=1,
            equity_curve=equity_curve,
            returns=portfolio_returns
        )


def print_backtest_results(results: BacktestResults, name: str = "Strategy"):
    """Print formatted backtest results."""
    print(f"\n{'='*50}")
    print(f" {name} Backtest Results")
    print(f"{'='*50}")
    print(f" Total Return:     {results.total_return:>10.2%}")
    print(f" CAGR:             {results.cagr:>10.2%}")
    print(f" Sharpe Ratio:     {results.sharpe_ratio:>10.3f}")
    print(f" Sortino Ratio:    {results.sortino_ratio:>10.3f}")
    print(f" Max Drawdown:     {results.max_drawdown:>10.2%}")
    print(f" Calmar Ratio:     {results.calmar_ratio:>10.3f}")
    print(f" Win Rate:         {results.win_rate:>10.2%}")
    print(f" Profit Factor:    {results.profit_factor:>10.3f}")
    print(f" Number of Trades: {results.n_trades:>10d}")
    print(f"{'='*50}\n")

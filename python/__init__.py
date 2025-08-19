"""
C-Mamba: Channel Correlation Enhanced State Space Models for Trading

This module provides implementations for multivariate time series forecasting
using C-Mamba architecture, specifically designed for financial trading applications.

Modules:
    - cmamba_model: Core C-Mamba implementation
    - data_loader: Data loading from Bybit and Yahoo Finance
    - backtest: Backtesting framework for trading strategies
    - train: Training utilities and helpers
    - features: Technical indicators and feature engineering
"""

from .cmamba_model import CMamba, CMambaConfig, CMambaTrainer
from .data_loader import (
    MultiAssetDataLoader,
    BybitClient,
    YahooFinanceClient,
    CRYPTO_UNIVERSE,
    STOCK_UNIVERSE,
)
from .backtest import CMambaBacktester, BacktestResults
from .features import FeatureEngineer, FeatureConfig, create_feature_matrix

__all__ = [
    # Model
    "CMamba",
    "CMambaConfig",
    "CMambaTrainer",
    # Data loaders
    "MultiAssetDataLoader",
    "BybitClient",
    "YahooFinanceClient",
    "CRYPTO_UNIVERSE",
    "STOCK_UNIVERSE",
    # Backtesting
    "CMambaBacktester",
    "BacktestResults",
    # Feature engineering
    "FeatureEngineer",
    "FeatureConfig",
    "create_feature_matrix",
]

__version__ = "0.1.0"

"""
Data loader for C-Mamba model.

Supports loading multi-asset data from:
- Bybit API (cryptocurrency data)
- Yahoo Finance (stock data)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BYBIT_API_BASE = "https://api.bybit.com"

# Default cryptocurrency universe
CRYPTO_UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    "MATICUSDT",
    "LINKUSDT",
]

# Default stock universe (tech sector)
STOCK_UNIVERSE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "AMD",
    "INTC",
    "CRM",
]


@dataclass
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class BybitClient:
    """
    Client for Bybit API.

    Fetches historical kline (candlestick) data for cryptocurrency trading pairs.
    """

    def __init__(self, base_url: str = BYBIT_API_BASE):
        self.base_url = base_url
        if not HAS_REQUESTS:
            logger.warning("requests library not installed. API calls will not work.")

    def get_klines(
        self,
        symbol: str,
        interval: str = "D",
        limit: int = 200,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Candle]:
        """
        Fetch kline data from Bybit.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Time interval ("1", "5", "15", "60", "240", "D", "W")
            limit: Number of candles (max 1000)
            start: Start time
            end: End time

        Returns:
            List of Candle objects
        """
        if not HAS_REQUESTS:
            return self._generate_mock_data(symbol, limit)

        params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        if start:
            params["start"] = int(start.timestamp() * 1000)
        if end:
            params["end"] = int(end.timestamp() * 1000)

        try:
            response = requests.get(
                f"{self.base_url}/v5/market/kline",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if data.get("retCode") != 0:
                logger.warning(f"Bybit API error: {data.get('retMsg')}")
                return self._generate_mock_data(symbol, limit)

            candles = []
            for item in reversed(data.get("result", {}).get("list", [])):
                candles.append(Candle(
                    timestamp=datetime.fromtimestamp(int(item[0]) / 1000),
                    open=float(item[1]),
                    high=float(item[2]),
                    low=float(item[3]),
                    close=float(item[4]),
                    volume=float(item[5]),
                ))

            return candles

        except Exception as e:
            logger.warning(f"Failed to fetch data from Bybit: {e}")
            return self._generate_mock_data(symbol, limit)

    def _generate_mock_data(self, symbol: str, limit: int) -> List[Candle]:
        """Generate mock data for testing."""
        logger.info(f"Generating mock data for {symbol}")

        # Base prices for different assets
        base_prices = {
            "BTCUSDT": 45000,
            "ETHUSDT": 2500,
            "SOLUSDT": 100,
            "BNBUSDT": 300,
            "XRPUSDT": 0.5,
            "ADAUSDT": 0.4,
            "AVAXUSDT": 35,
            "DOTUSDT": 7,
            "MATICUSDT": 0.8,
            "LINKUSDT": 15,
        }

        base = base_prices.get(symbol, 100)
        volatility = 0.02

        candles = []
        price = base
        now = datetime.now()

        for i in range(limit):
            timestamp = now - timedelta(days=limit - i)

            # Random walk with mean reversion
            change = np.random.normal(0, volatility)
            price = price * (1 + change)

            # Add some correlation to BTC for other assets
            if symbol != "BTCUSDT":
                btc_influence = np.random.normal(0, volatility * 0.5)
                price = price * (1 + btc_influence)

            candles.append(Candle(
                timestamp=timestamp,
                open=price,
                high=price * (1 + abs(np.random.normal(0, volatility))),
                low=price * (1 - abs(np.random.normal(0, volatility))),
                close=price * (1 + np.random.normal(0, volatility * 0.5)),
                volume=np.random.uniform(1e6, 1e8),
            ))

        return candles

    def fetch_multi_asset(
        self,
        symbols: List[str],
        interval: str = "D",
        limit: int = 200
    ) -> Dict[str, List[Candle]]:
        """
        Fetch data for multiple assets.

        Args:
            symbols: List of trading pairs
            interval: Time interval
            limit: Number of candles per asset

        Returns:
            Dictionary mapping symbol to list of candles
        """
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_klines(symbol, interval, limit)
            logger.info(f"Fetched {len(data[symbol])} candles for {symbol}")
        return data


class YahooFinanceClient:
    """
    Client for Yahoo Finance data.

    Fetches historical stock data using the yfinance library.
    """

    def __init__(self):
        if not HAS_YFINANCE:
            logger.warning("yfinance library not installed. Yahoo Finance API calls will not work.")

    def get_klines(
        self,
        symbol: str,
        interval: str = "D",
        limit: int = 200,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[Candle]:
        """
        Fetch kline (OHLCV) data from Yahoo Finance.

        Args:
            symbol: Stock ticker (e.g., "AAPL", "MSFT")
            interval: Time interval ("D" for daily, "W" for weekly)
            limit: Number of candles
            start: Start time
            end: End time

        Returns:
            List of Candle objects
        """
        if not HAS_YFINANCE:
            return self._generate_mock_data(symbol, limit)

        try:
            ticker = yf.Ticker(symbol)

            # Calculate date range
            if end is None:
                end = datetime.now()
            if start is None:
                start = end - timedelta(days=limit * 2)  # Extra buffer for non-trading days

            # Map interval
            interval_map = {"D": "1d", "W": "1wk", "M": "1mo"}
            yf_interval = interval_map.get(interval, "1d")

            # Fetch data
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=yf_interval
            )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return self._generate_mock_data(symbol, limit)

            candles = []
            for idx, row in df.iterrows():
                candles.append(Candle(
                    timestamp=idx.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                ))

            # Limit to requested number
            if len(candles) > limit:
                candles = candles[-limit:]

            return candles

        except Exception as e:
            logger.warning(f"Failed to fetch data from Yahoo Finance: {e}")
            return self._generate_mock_data(symbol, limit)

    def _generate_mock_data(self, symbol: str, limit: int) -> List[Candle]:
        """Generate mock stock data for testing."""
        logger.info(f"Generating mock data for {symbol}")

        # Base prices for different stocks
        base_prices = {
            "AAPL": 180,
            "MSFT": 380,
            "GOOGL": 140,
            "AMZN": 180,
            "META": 500,
            "NVDA": 700,
            "TSLA": 250,
            "AMD": 160,
            "INTC": 45,
            "CRM": 280,
        }

        base = base_prices.get(symbol, 100)
        volatility = 0.015  # Stocks typically less volatile than crypto

        candles = []
        price = base
        now = datetime.now()

        for i in range(limit):
            timestamp = now - timedelta(days=limit - i)

            # Skip weekends (simple approximation)
            if timestamp.weekday() >= 5:
                continue

            # Random walk with drift
            change = np.random.normal(0.0003, volatility)  # Slight upward drift
            price = price * (1 + change)

            candles.append(Candle(
                timestamp=timestamp,
                open=price,
                high=price * (1 + abs(np.random.normal(0, volatility))),
                low=price * (1 - abs(np.random.normal(0, volatility))),
                close=price * (1 + np.random.normal(0, volatility * 0.5)),
                volume=np.random.uniform(1e7, 1e9),
            ))

        return candles

    def fetch_multi_asset(
        self,
        symbols: List[str],
        interval: str = "D",
        limit: int = 200
    ) -> Dict[str, List[Candle]]:
        """
        Fetch data for multiple stock tickers.

        Args:
            symbols: List of stock tickers
            interval: Time interval
            limit: Number of candles per asset

        Returns:
            Dictionary mapping symbol to list of candles
        """
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_klines(symbol, interval, limit)
            logger.info(f"Fetched {len(data[symbol])} candles for {symbol}")
        return data


class MultiAssetDataLoader:
    """
    Multi-asset data loader for C-Mamba.

    Loads and preprocesses data from multiple sources for training and inference.
    Supports:
        - Bybit API for cryptocurrency data
        - Yahoo Finance for stock data
    """

    def __init__(
        self,
        symbols: List[str] = None,
        source: str = "bybit",
        interval: str = "D"
    ):
        self.source = source
        self.interval = interval

        if source == "bybit":
            self.symbols = symbols or CRYPTO_UNIVERSE
            self.client = BybitClient()
        elif source == "yahoo":
            self.symbols = symbols or STOCK_UNIVERSE
            self.client = YahooFinanceClient()
        else:
            raise ValueError(f"Unsupported source: {source}. Use 'bybit' or 'yahoo'")

    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 365
    ) -> Tuple[np.ndarray, List[datetime]]:
        """
        Load multi-asset data.

        Args:
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            days: Number of days if dates not specified

        Returns:
            Tuple of (data array [timesteps, n_channels], timestamps)
        """
        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end - start).days

        # Fetch data for all symbols
        all_data = self.client.fetch_multi_asset(self.symbols, self.interval, days)

        # Align data by timestamp and create matrix
        return self._align_and_create_matrix(all_data)

    def _align_and_create_matrix(
        self,
        all_data: Dict[str, List[Candle]]
    ) -> Tuple[np.ndarray, List[datetime]]:
        """
        Align data across assets and create feature matrix.

        Args:
            all_data: Dictionary of symbol -> candles

        Returns:
            Tuple of (data array, timestamps)
        """
        # Find common timestamps
        timestamps_sets = [
            set(c.timestamp for c in candles)
            for candles in all_data.values()
        ]
        common_timestamps = sorted(set.intersection(*timestamps_sets))

        if not common_timestamps:
            # Fallback: use first symbol's timestamps
            first_symbol = list(all_data.keys())[0]
            common_timestamps = [c.timestamp for c in all_data[first_symbol]]

        n_timesteps = len(common_timestamps)
        n_channels = len(self.symbols)

        # Create price matrix (using close prices)
        prices = np.zeros((n_timesteps, n_channels))

        for j, symbol in enumerate(self.symbols):
            candles = all_data[symbol]
            price_dict = {c.timestamp: c.close for c in candles}

            for i, ts in enumerate(common_timestamps):
                if ts in price_dict:
                    prices[i, j] = price_dict[ts]
                elif i > 0:
                    prices[i, j] = prices[i-1, j]  # Forward fill
                else:
                    prices[i, j] = candles[0].close if candles else 0

        return prices, common_timestamps

    def load_and_split(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data and split into train/val/test sets.

        Args:
            start_date: Start date
            end_date: End date
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            normalize: Whether to normalize the data

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        data, timestamps = self.load_data(start_date, end_date)

        # Normalize
        if normalize:
            data = self._normalize(data)

        # Split
        n = len(data)
        test_start = int(n * (1 - test_ratio))
        val_start = int(n * (1 - test_ratio - val_ratio))

        train_data = data[:val_start]
        val_data = data[val_start:test_start]
        test_data = data[test_start:]

        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        return train_data, val_data, test_data

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using returns.

        Args:
            data: Price data [timesteps, n_channels]

        Returns:
            Normalized returns [timesteps-1, n_channels]
        """
        # Calculate log returns
        returns = np.diff(np.log(data + 1e-8), axis=0)

        # Standardize
        mean = returns.mean(axis=0, keepdims=True)
        std = returns.std(axis=0, keepdims=True) + 1e-8

        return (returns - mean) / std

    def prepare_sequences(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input-output sequences for training.

        Args:
            data: Time series data [timesteps, n_channels]
            seq_len: Input sequence length
            pred_len: Prediction length

        Returns:
            Tuple of (X, Y) where:
            - X: [n_samples, seq_len, n_channels]
            - Y: [n_samples, pred_len, n_channels]
        """
        n_samples = len(data) - seq_len - pred_len + 1

        X = np.zeros((n_samples, seq_len, data.shape[1]))
        Y = np.zeros((n_samples, pred_len, data.shape[1]))

        for i in range(n_samples):
            X[i] = data[i:i+seq_len]
            Y[i] = data[i+seq_len:i+seq_len+pred_len]

        return X, Y


def calculate_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Calculate correlation matrix for multi-asset data.

    Args:
        data: Price data [timesteps, n_channels]

    Returns:
        Correlation matrix [n_channels, n_channels]
    """
    # Calculate returns
    returns = np.diff(data, axis=0) / (data[:-1] + 1e-8)

    # Correlation matrix
    return np.corrcoef(returns.T)


def identify_correlation_groups(
    corr_matrix: np.ndarray,
    symbols: List[str],
    threshold: float = 0.7
) -> Dict[str, List[str]]:
    """
    Identify groups of highly correlated assets.

    Args:
        corr_matrix: Correlation matrix
        symbols: List of symbol names
        threshold: Correlation threshold for grouping

    Returns:
        Dictionary of group name -> list of symbols
    """
    n = len(symbols)
    visited = [False] * n
    groups = {}
    group_id = 0

    for i in range(n):
        if visited[i]:
            continue

        # Start new group
        group = [symbols[i]]
        visited[i] = True

        # Find correlated assets
        for j in range(i + 1, n):
            if not visited[j] and abs(corr_matrix[i, j]) >= threshold:
                group.append(symbols[j])
                visited[j] = True

        groups[f"Group_{group_id}"] = group
        group_id += 1

    return groups

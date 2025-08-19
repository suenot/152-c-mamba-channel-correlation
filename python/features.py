"""
Feature engineering module for C-Mamba trading model.

Provides technical indicators, statistical features, and data transformations
for multivariate time series analysis in financial markets.

Features included:
- Price-based indicators (returns, log returns)
- Momentum indicators (RSI, MACD, ROC)
- Volatility indicators (ATR, Bollinger Bands, Keltner Channels)
- Volume indicators (OBV, VWAP, MFI)
- Statistical features (rolling stats, z-scores)
- Cross-asset features (correlation, spread)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Window sizes for rolling calculations
    short_window: int = 5
    medium_window: int = 20
    long_window: int = 60

    # RSI
    rsi_period: int = 14

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR
    atr_period: int = 14

    # Include flags
    include_returns: bool = True
    include_momentum: bool = True
    include_volatility: bool = True
    include_volume: bool = True
    include_stats: bool = True


class FeatureEngineer:
    """
    Feature engineering for financial time series.

    Computes technical indicators and statistical features for trading models.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()

    def compute_all_features(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray] = None,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute all configured features.

        Args:
            prices: Close prices [timesteps, n_assets]
            volumes: Volume data [timesteps, n_assets] (optional)
            high: High prices [timesteps, n_assets] (optional)
            low: Low prices [timesteps, n_assets] (optional)

        Returns:
            Dictionary of feature arrays
        """
        features = {}

        if self.config.include_returns:
            features.update(self.compute_returns(prices))

        if self.config.include_momentum:
            features.update(self.compute_momentum(prices))

        if self.config.include_volatility:
            features.update(self.compute_volatility(prices, high, low))

        if self.config.include_volume and volumes is not None:
            features.update(self.compute_volume_features(prices, volumes))

        if self.config.include_stats:
            features.update(self.compute_statistics(prices))

        return features

    def compute_returns(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute return-based features.

        Args:
            prices: Close prices [timesteps, n_assets]

        Returns:
            Dictionary with return features
        """
        features = {}

        # Simple returns
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)
        features["returns"] = returns

        # Log returns
        log_returns = np.zeros_like(prices)
        log_returns[1:] = np.log(prices[1:] / (prices[:-1] + 1e-8))
        features["log_returns"] = log_returns

        # Cumulative returns over windows
        for window in [self.config.short_window, self.config.medium_window]:
            cum_returns = self._rolling_sum(log_returns, window)
            features[f"cum_returns_{window}"] = cum_returns

        return features

    def compute_momentum(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute momentum indicators.

        Args:
            prices: Close prices [timesteps, n_assets]

        Returns:
            Dictionary with momentum features
        """
        features = {}

        # RSI
        features["rsi"] = self._compute_rsi(prices, self.config.rsi_period)

        # MACD
        macd, signal, histogram = self._compute_macd(
            prices,
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_histogram"] = histogram

        # Rate of Change (ROC)
        for period in [self.config.short_window, self.config.medium_window]:
            roc = np.zeros_like(prices)
            roc[period:] = (prices[period:] - prices[:-period]) / (prices[:-period] + 1e-8)
            features[f"roc_{period}"] = roc

        # Momentum
        momentum = np.zeros_like(prices)
        momentum[self.config.medium_window:] = prices[self.config.medium_window:] - prices[:-self.config.medium_window]
        features["momentum"] = momentum

        return features

    def compute_volatility(
        self,
        prices: np.ndarray,
        high: Optional[np.ndarray] = None,
        low: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute volatility indicators.

        Args:
            prices: Close prices [timesteps, n_assets]
            high: High prices (optional)
            low: Low prices (optional)

        Returns:
            Dictionary with volatility features
        """
        features = {}

        # Rolling volatility (standard deviation of returns)
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

        for window in [self.config.short_window, self.config.medium_window, self.config.long_window]:
            vol = self._rolling_std(returns, window)
            features[f"volatility_{window}"] = vol

        # Bollinger Bands
        bb_mid, bb_upper, bb_lower = self._compute_bollinger_bands(
            prices, self.config.bb_period, self.config.bb_std
        )
        features["bb_mid"] = bb_mid
        features["bb_upper"] = bb_upper
        features["bb_lower"] = bb_lower
        features["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-8)
        features["bb_position"] = (prices - bb_lower) / (bb_upper - bb_lower + 1e-8)

        # ATR (Average True Range) if high/low available
        if high is not None and low is not None:
            atr = self._compute_atr(high, low, prices, self.config.atr_period)
            features["atr"] = atr
            features["atr_percent"] = atr / (prices + 1e-8)

        return features

    def compute_volume_features(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute volume-based features.

        Args:
            prices: Close prices [timesteps, n_assets]
            volumes: Volume data [timesteps, n_assets]

        Returns:
            Dictionary with volume features
        """
        features = {}

        # Volume change
        vol_change = np.zeros_like(volumes)
        vol_change[1:] = (volumes[1:] - volumes[:-1]) / (volumes[:-1] + 1e-8)
        features["volume_change"] = vol_change

        # Volume moving average ratio
        vol_ma = self._rolling_mean(volumes, self.config.medium_window)
        features["volume_ma_ratio"] = volumes / (vol_ma + 1e-8)

        # OBV (On-Balance Volume)
        features["obv"] = self._compute_obv(prices, volumes)

        # VWAP approximation (using rolling window)
        features["vwap"] = self._compute_vwap(prices, volumes, self.config.medium_window)

        return features

    def compute_statistics(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute statistical features.

        Args:
            prices: Close prices [timesteps, n_assets]

        Returns:
            Dictionary with statistical features
        """
        features = {}

        # Z-score normalization
        for window in [self.config.short_window, self.config.medium_window]:
            z_score = self._compute_zscore(prices, window)
            features[f"zscore_{window}"] = z_score

        # Rolling skewness and kurtosis
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

        features["skewness"] = self._rolling_skewness(returns, self.config.medium_window)
        features["kurtosis"] = self._rolling_kurtosis(returns, self.config.medium_window)

        # Price relative to moving averages
        for window in [self.config.short_window, self.config.medium_window, self.config.long_window]:
            ma = self._rolling_mean(prices, window)
            features[f"price_ma_ratio_{window}"] = prices / (ma + 1e-8)

        return features

    def compute_cross_asset_features(
        self,
        prices: np.ndarray,
        asset_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute cross-asset correlation features.

        Args:
            prices: Close prices [timesteps, n_assets]
            asset_names: Optional list of asset names

        Returns:
            Dictionary with cross-asset features
        """
        features = {}
        n_assets = prices.shape[1]

        # Rolling correlation matrix
        window = self.config.medium_window
        n_timesteps = len(prices)

        # Compute returns
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)

        # Rolling correlations (average cross-correlation)
        avg_corr = np.zeros(n_timesteps)
        for t in range(window, n_timesteps):
            window_returns = returns[t - window:t]
            if window_returns.shape[0] >= 2:
                corr_matrix = np.corrcoef(window_returns.T)
                # Average of upper triangle (excluding diagonal)
                upper_tri = corr_matrix[np.triu_indices(n_assets, k=1)]
                avg_corr[t] = np.nanmean(upper_tri)

        features["avg_cross_correlation"] = avg_corr

        # Relative strength vs market average
        market_return = np.mean(returns, axis=1, keepdims=True)
        features["relative_strength"] = returns - market_return

        # Spread features (each asset vs mean)
        mean_price = np.mean(prices, axis=1, keepdims=True)
        features["spread_from_mean"] = (prices - mean_price) / (mean_price + 1e-8)

        return features

    # ========== Helper Methods ==========

    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean."""
        result = np.zeros_like(data)
        for i in range(window - 1, len(data)):
            result[i] = np.mean(data[i - window + 1:i + 1], axis=0)
        return result

    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation."""
        result = np.zeros_like(data)
        for i in range(window - 1, len(data)):
            result[i] = np.std(data[i - window + 1:i + 1], axis=0)
        return result

    def _rolling_sum(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling sum."""
        result = np.zeros_like(data)
        for i in range(window - 1, len(data)):
            result[i] = np.sum(data[i - window + 1:i + 1], axis=0)
        return result

    def _rolling_skewness(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling skewness."""
        result = np.zeros_like(data)
        for i in range(window - 1, len(data)):
            window_data = data[i - window + 1:i + 1]
            mean = np.mean(window_data, axis=0)
            std = np.std(window_data, axis=0) + 1e-8
            result[i] = np.mean(((window_data - mean) / std) ** 3, axis=0)
        return result

    def _rolling_kurtosis(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling kurtosis."""
        result = np.zeros_like(data)
        for i in range(window - 1, len(data)):
            window_data = data[i - window + 1:i + 1]
            mean = np.mean(window_data, axis=0)
            std = np.std(window_data, axis=0) + 1e-8
            result[i] = np.mean(((window_data - mean) / std) ** 4, axis=0) - 3
        return result

    def _compute_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Compute Relative Strength Index."""
        result = np.zeros_like(prices)

        # Calculate price changes
        delta = np.zeros_like(prices)
        delta[1:] = prices[1:] - prices[:-1]

        # Separate gains and losses
        gains = np.maximum(delta, 0)
        losses = np.maximum(-delta, 0)

        # Calculate RSI
        for i in range(period, len(prices)):
            avg_gain = np.mean(gains[i - period + 1:i + 1], axis=0)
            avg_loss = np.mean(losses[i - period + 1:i + 1], axis=0) + 1e-8
            rs = avg_gain / avg_loss
            result[i] = 100 - (100 / (1 + rs))

        return result

    def _compute_macd(
        self,
        prices: np.ndarray,
        fast: int,
        slow: int,
        signal: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute MACD indicator."""
        # EMA calculation
        def ema(data, period):
            result = np.zeros_like(data)
            alpha = 2 / (period + 1)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            return result

        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _compute_bollinger_bands(
        self,
        prices: np.ndarray,
        period: int,
        std_dev: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Bollinger Bands."""
        middle = self._rolling_mean(prices, period)
        std = self._rolling_std(prices, period)

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        return middle, upper, lower

    def _compute_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Compute Average True Range."""
        # True Range
        tr = np.zeros_like(close)
        tr[1:] = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        tr[0] = high[0] - low[0]

        # ATR is smoothed TR
        return self._rolling_mean(tr, period)

    def _compute_obv(
        self,
        prices: np.ndarray,
        volumes: np.ndarray
    ) -> np.ndarray:
        """Compute On-Balance Volume."""
        obv = np.zeros_like(prices)

        for i in range(1, len(prices)):
            sign = np.sign(prices[i] - prices[i - 1])
            obv[i] = obv[i - 1] + sign * volumes[i]

        return obv

    def _compute_vwap(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        window: int
    ) -> np.ndarray:
        """Compute rolling VWAP."""
        result = np.zeros_like(prices)

        for i in range(window - 1, len(prices)):
            window_prices = prices[i - window + 1:i + 1]
            window_volumes = volumes[i - window + 1:i + 1]
            total_volume = np.sum(window_volumes, axis=0) + 1e-8
            result[i] = np.sum(window_prices * window_volumes, axis=0) / total_volume

        return result

    def _compute_zscore(self, data: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling z-score."""
        mean = self._rolling_mean(data, window)
        std = self._rolling_std(data, window) + 1e-8
        return (data - mean) / std


def create_feature_matrix(
    prices: np.ndarray,
    volumes: Optional[np.ndarray] = None,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    config: Optional[FeatureConfig] = None,
    include_cross_asset: bool = True
) -> np.ndarray:
    """
    Create a complete feature matrix from OHLCV data.

    Args:
        prices: Close prices [timesteps, n_assets]
        volumes: Volume data (optional)
        high: High prices (optional)
        low: Low prices (optional)
        config: Feature configuration
        include_cross_asset: Whether to include cross-asset features

    Returns:
        Feature matrix [timesteps, n_assets * n_features]
    """
    engineer = FeatureEngineer(config)

    # Compute all features
    features = engineer.compute_all_features(prices, volumes, high, low)

    if include_cross_asset:
        cross_features = engineer.compute_cross_asset_features(prices)
        features.update(cross_features)

    # Stack features
    feature_list = []
    for name, arr in sorted(features.items()):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        feature_list.append(arr)

    feature_matrix = np.concatenate(feature_list, axis=1)

    logger.info(f"Created feature matrix with shape {feature_matrix.shape}")
    logger.info(f"Features: {list(features.keys())}")

    return feature_matrix


def normalize_features(
    features: np.ndarray,
    method: str = "zscore",
    clip: Optional[float] = 3.0
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize feature matrix.

    Args:
        features: Feature matrix [timesteps, n_features]
        method: Normalization method ('zscore', 'minmax', 'robust')
        clip: Clip extreme values (for zscore)

    Returns:
        Tuple of (normalized features, normalization parameters)
    """
    params = {}

    if method == "zscore":
        mean = np.nanmean(features, axis=0)
        std = np.nanstd(features, axis=0) + 1e-8
        normalized = (features - mean) / std

        if clip is not None:
            normalized = np.clip(normalized, -clip, clip)

        params = {"mean": mean, "std": std}

    elif method == "minmax":
        min_val = np.nanmin(features, axis=0)
        max_val = np.nanmax(features, axis=0)
        range_val = max_val - min_val + 1e-8
        normalized = (features - min_val) / range_val

        params = {"min": min_val, "max": max_val}

    elif method == "robust":
        median = np.nanmedian(features, axis=0)
        q75 = np.nanpercentile(features, 75, axis=0)
        q25 = np.nanpercentile(features, 25, axis=0)
        iqr = q75 - q25 + 1e-8
        normalized = (features - median) / iqr

        if clip is not None:
            normalized = np.clip(normalized, -clip, clip)

        params = {"median": median, "iqr": iqr}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_timesteps = 500
    n_assets = 5

    # Simulate price data
    prices = np.zeros((n_timesteps, n_assets))
    prices[0] = [100, 50, 200, 75, 150]  # Initial prices

    for t in range(1, n_timesteps):
        returns = np.random.normal(0.0005, 0.02, n_assets)
        prices[t] = prices[t - 1] * (1 + returns)

    # Simulate volume data
    volumes = np.random.uniform(1e6, 1e8, (n_timesteps, n_assets))

    # Create features
    config = FeatureConfig(
        short_window=5,
        medium_window=20,
        long_window=60
    )

    feature_matrix = create_feature_matrix(
        prices=prices,
        volumes=volumes,
        config=config,
        include_cross_asset=True
    )

    print(f"Feature matrix shape: {feature_matrix.shape}")

    # Normalize
    normalized, params = normalize_features(feature_matrix, method="zscore")
    print(f"Normalized features shape: {normalized.shape}")
    print(f"Mean of normalized features: {np.nanmean(normalized):.4f}")
    print(f"Std of normalized features: {np.nanstd(normalized):.4f}")

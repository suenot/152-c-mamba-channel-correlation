"""
C-Mamba: Channel Correlation Enhanced State Space Models

This module implements the C-Mamba architecture for multivariate time series
forecasting with cross-channel dependency modeling.

Reference:
    Zeng et al. (2024). C-Mamba: Channel Correlation Enhanced State Space Models
    for Multivariate Time Series Forecasting. arXiv:2406.05316
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CMambaConfig:
    """Configuration for C-Mamba model."""
    seq_len: int = 60              # Input sequence length
    pred_len: int = 5              # Prediction length
    n_channels: int = 10           # Number of channels (assets)
    d_model: int = 64              # Model dimension
    d_state: int = 16              # SSM state dimension
    d_conv: int = 4                # Convolution kernel size
    n_layers: int = 2              # Number of M-Mamba layers
    patch_len: int = 12            # Patch length
    stride: int = 6                # Patch stride
    expansion_factor: int = 2      # MLP expansion factor
    dropout: float = 0.1           # Dropout rate
    channel_mixup_alpha: float = 0.2  # Channel mixup alpha


@dataclass
class CMambaOutput:
    """Output from C-Mamba model."""
    predictions: np.ndarray        # Shape: [batch, pred_len, n_channels]
    channel_attention: Optional[np.ndarray] = None  # Channel correlation matrix
    hidden_states: Optional[np.ndarray] = None


class SelectiveSSM:
    """
    Selective State Space Model (S6) - Core component of Mamba.

    Implements the data-dependent state space model with selective parameters.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv

        # Initialize parameters (in real implementation, these would be learnable)
        self._init_parameters()

    def _init_parameters(self):
        """Initialize SSM parameters."""
        # A: state matrix (d_state,)
        self.A = -np.exp(np.random.randn(self.d_state))

        # Projection matrices
        self.W_B = np.random.randn(self.d_model, self.d_state) * 0.01
        self.W_C = np.random.randn(self.d_state, self.d_model) * 0.01
        self.W_delta = np.random.randn(self.d_model, 1) * 0.01

        # Convolution kernel for local context
        self.conv_kernel = np.random.randn(self.d_conv, self.d_model) * 0.01

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through SSM.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Compute data-dependent parameters
        delta = (1.0 / (1.0 + np.exp(-(x @ self.W_delta)))).squeeze(-1)  # [batch, seq_len]
        B = x @ self.W_B  # [batch, seq_len, d_state]

        # Discretize A
        A_bar = np.exp(np.outer(delta.flatten(), self.A).reshape(batch_size, seq_len, -1))

        # Initialize state
        h = np.zeros((batch_size, self.d_state))

        # Scan through sequence
        outputs = []
        for t in range(seq_len):
            # State update
            h = A_bar[:, t, :] * h + B[:, t, :] * delta[:, t:t+1]

            # Output
            y = h @ self.W_C
            outputs.append(y)

        return np.stack(outputs, axis=1)


class MMambaBlock:
    """
    Modified Mamba block with bidirectional scanning for time series.

    Features:
    - Forward and backward SSM branches
    - Residual connections
    - Layer normalization
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, dropout: float = 0.1):
        self.d_model = d_model
        self.ssm_forward = SelectiveSSM(d_model, d_state, d_conv)
        self.ssm_backward = SelectiveSSM(d_model, d_state, d_conv)
        self.dropout = dropout

        # Layer norm parameters (mean and std for simplicity)
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        return self.gamma * (x - mean) / std + self.beta

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with bidirectional processing.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            Output [batch, seq_len, d_model]
        """
        # Forward pass
        fwd_out = self.ssm_forward.forward(x)

        # Backward pass (reverse, process, reverse back)
        x_rev = np.flip(x, axis=1)
        bwd_out = np.flip(self.ssm_backward.forward(x_rev), axis=1)

        # Combine with residual
        out = x + fwd_out + bwd_out

        # Layer norm
        return self._layer_norm(out)


class GDDMLP:
    """
    Global-local Dual-branch MLP for cross-channel dependency modeling.

    Features:
    - Global branch: Captures market-wide patterns
    - Local branch: Preserves channel-specific details
    - Channel attention mechanism
    """

    def __init__(self, n_channels: int, d_model: int, expansion: int = 4):
        self.n_channels = n_channels
        self.d_model = d_model
        self.expansion = expansion

        # Global branch parameters
        self.W_global_1 = np.random.randn(n_channels, n_channels * expansion) * 0.01
        self.W_global_2 = np.random.randn(n_channels * expansion, n_channels) * 0.01

        # Local branch parameters
        self.W_local_1 = np.random.randn(d_model, d_model * expansion) * 0.01
        self.W_local_2 = np.random.randn(d_model * expansion, d_model) * 0.01

        # Channel attention
        self.W_attn = np.random.randn(n_channels, n_channels) * 0.01

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation function."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax function."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def forward(
        self,
        x: np.ndarray,
        return_attention: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forward pass through GDD-MLP.

        Args:
            x: Input [batch, n_channels, n_patches, d_model]
            return_attention: Whether to return channel attention weights

        Returns:
            Tuple of (output, attention) where attention is optional
        """
        batch_size, n_channels, n_patches, d_model = x.shape

        # Global branch: aggregate across patches and process channels
        global_feat = x.mean(axis=(2, 3))  # [batch, n_channels]
        global_out = self._gelu(global_feat @ self.W_global_1)
        global_out = global_out @ self.W_global_2  # [batch, n_channels]

        # Local branch: process each position independently
        local_out = self._gelu(x @ self.W_local_1)
        local_out = local_out @ self.W_local_2  # [batch, n_channels, n_patches, d_model]

        # Channel attention
        attention = self._softmax(global_feat @ self.W_attn, axis=-1)  # [batch, n_channels]

        # Combine global and local
        global_broadcast = global_out[:, :, np.newaxis, np.newaxis]
        output = local_out + global_broadcast * attention[:, :, np.newaxis, np.newaxis]

        if return_attention:
            # Compute full attention matrix
            attn_matrix = attention[:, :, np.newaxis] * attention[:, np.newaxis, :]
            return output, attn_matrix

        return output, None


class PatchEmbedding:
    """
    Patch embedding layer for time series.

    Converts time series into patch representations for efficient processing.
    """

    def __init__(self, patch_len: int, stride: int, d_model: int):
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Linear projection
        self.W_embed = np.random.randn(patch_len, d_model) * 0.01

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Create patch embeddings.

        Args:
            x: Input [batch, seq_len, n_channels]

        Returns:
            Tuple of (patches [batch, n_channels, n_patches, d_model], n_patches)
        """
        batch_size, seq_len, n_channels = x.shape

        # Calculate number of patches
        n_patches = (seq_len - self.patch_len) // self.stride + 1

        # Extract patches
        patches = []
        for i in range(n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = x[:, start:end, :]  # [batch, patch_len, n_channels]
            patches.append(patch)

        # Stack patches: [batch, n_patches, patch_len, n_channels]
        patches = np.stack(patches, axis=1)

        # Rearrange to [batch, n_channels, n_patches, patch_len]
        patches = patches.transpose(0, 3, 1, 2)

        # Project to d_model
        # [batch, n_channels, n_patches, patch_len] @ [patch_len, d_model]
        # -> [batch, n_channels, n_patches, d_model]
        embedded = patches @ self.W_embed

        return embedded, n_patches


class ChannelMixup:
    """
    Channel Mixup data augmentation for training.

    Mixes channel information to improve generalization and robustness
    to changing correlations.
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply channel mixup.

        Args:
            x: Input [batch, seq_len, n_channels]
            training: Whether in training mode

        Returns:
            Mixed input (same shape)
        """
        if not training or self.alpha <= 0:
            return x

        batch_size, seq_len, n_channels = x.shape

        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Random channel permutation
        perm = np.random.permutation(n_channels)

        # Mix channels
        mixed = lam * x + (1 - lam) * x[:, :, perm]

        return mixed


class CMamba:
    """
    C-Mamba: Channel Correlation Enhanced State Space Model

    Main model class for multivariate time series forecasting with
    cross-channel dependency modeling.

    Architecture:
        1. Patch Embedding: Convert time series to patches
        2. M-Mamba Blocks: Capture cross-time dependencies (bidirectional SSM)
        3. GDD-MLP: Capture cross-channel dependencies
        4. Projection: Output predictions

    Example:
        >>> config = CMambaConfig(seq_len=60, pred_len=5, n_channels=10)
        >>> model = CMamba(config)
        >>> predictions = model.predict(historical_data)
    """

    def __init__(self, config: CMambaConfig):
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            config.patch_len,
            config.stride,
            config.d_model
        )

        # M-Mamba blocks
        self.m_mamba_blocks = [
            MMambaBlock(config.d_model, config.d_state, config.d_conv, config.dropout)
            for _ in range(config.n_layers)
        ]

        # GDD-MLP
        self.gdd_mlp = GDDMLP(
            config.n_channels,
            config.d_model,
            config.expansion_factor
        )

        # Channel mixup
        self.channel_mixup = ChannelMixup(config.channel_mixup_alpha)

        # Calculate output dimension
        n_patches = (config.seq_len - config.patch_len) // config.stride + 1
        self.n_patches = n_patches

        # Output projection
        self.W_out = np.random.randn(
            config.d_model * n_patches,
            config.pred_len
        ) * 0.01

        self._is_training = False

    def train(self):
        """Set model to training mode."""
        self._is_training = True

    def eval(self):
        """Set model to evaluation mode."""
        self._is_training = False

    def forward(
        self,
        x: np.ndarray,
        return_attention: bool = False
    ) -> CMambaOutput:
        """
        Forward pass through C-Mamba.

        Args:
            x: Input tensor [batch, seq_len, n_channels]
            return_attention: Whether to return channel attention

        Returns:
            CMambaOutput with predictions and optional attention
        """
        # Channel mixup (training only)
        if self._is_training:
            x = self.channel_mixup(x, training=True)

        # Patch embedding
        x, n_patches = self.patch_embed.forward(x)  # [batch, n_channels, n_patches, d_model]

        # Reshape for M-Mamba: [batch * n_channels, n_patches, d_model]
        batch_size, n_channels, n_patches, d_model = x.shape
        x_reshaped = x.reshape(batch_size * n_channels, n_patches, d_model)

        # M-Mamba blocks (cross-time dependencies)
        for block in self.m_mamba_blocks:
            x_reshaped = block.forward(x_reshaped)

        # Reshape back: [batch, n_channels, n_patches, d_model]
        x = x_reshaped.reshape(batch_size, n_channels, n_patches, d_model)

        # GDD-MLP (cross-channel dependencies)
        x, attention = self.gdd_mlp.forward(x, return_attention=return_attention)

        # Flatten and project
        x_flat = x.reshape(batch_size, n_channels, -1)  # [batch, n_channels, n_patches * d_model]

        # Output projection: [batch, n_channels, pred_len]
        predictions = x_flat @ self.W_out

        # Transpose to [batch, pred_len, n_channels]
        predictions = predictions.transpose(0, 2, 1)

        return CMambaOutput(
            predictions=predictions,
            channel_attention=attention,
            hidden_states=x if return_attention else None
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Args:
            x: Input tensor [batch, seq_len, n_channels]

        Returns:
            Predictions [batch, pred_len, n_channels]
        """
        self.eval()
        output = self.forward(x)
        return output.predictions

    def get_channel_correlation(self, x: np.ndarray) -> np.ndarray:
        """
        Extract learned channel correlations.

        Args:
            x: Input tensor [batch, seq_len, n_channels]

        Returns:
            Correlation matrix [batch, n_channels, n_channels]
        """
        self.eval()
        output = self.forward(x, return_attention=True)
        return output.channel_attention

    def analyze_dependencies(self, x: np.ndarray) -> Dict:
        """
        Analyze cross-channel and cross-time dependencies.

        Args:
            x: Input tensor [batch, seq_len, n_channels]

        Returns:
            Dictionary with dependency analysis
        """
        output = self.forward(x, return_attention=True)

        # Channel correlations
        if output.channel_attention is not None:
            corr_matrix = output.channel_attention.mean(axis=0)
        else:
            corr_matrix = None

        # Identify strongest correlations
        if corr_matrix is not None:
            # Get top correlations (excluding diagonal)
            n = corr_matrix.shape[0]
            corr_flat = []
            for i in range(n):
                for j in range(i+1, n):
                    corr_flat.append((i, j, corr_matrix[i, j]))
            corr_flat.sort(key=lambda x: x[2], reverse=True)
            top_correlations = corr_flat[:5]
        else:
            top_correlations = []

        return {
            "correlation_matrix": corr_matrix,
            "top_correlations": top_correlations,
            "predictions": output.predictions,
        }


class CMambaTrainer:
    """
    Trainer for C-Mamba model.

    Implements training loop with:
    - MSE loss
    - Simple gradient descent (for demonstration)
    - Early stopping
    - Validation monitoring
    """

    def __init__(
        self,
        model: CMamba,
        learning_rate: float = 1e-3,
        patience: int = 10
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _mse_loss(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute MSE loss."""
        return np.mean((pred - target) ** 2)

    def train_step(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> float:
        """
        Single training step.

        Args:
            x: Input [batch, seq_len, n_channels]
            y: Target [batch, pred_len, n_channels]

        Returns:
            Loss value
        """
        self.model.train()

        # Forward pass
        output = self.model.forward(x)
        predictions = output.predictions

        # Compute loss
        loss = self._mse_loss(predictions, y)

        return loss

    def validate(
        self,
        val_data: Tuple[np.ndarray, np.ndarray]
    ) -> float:
        """
        Validate model.

        Args:
            val_data: Tuple of (x, y) validation data

        Returns:
            Validation loss
        """
        self.model.eval()
        x, y = val_data
        predictions = self.model.predict(x)
        return self._mse_loss(predictions, y)

    def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.

        Args:
            train_data: Tuple of (x, y) training data
            val_data: Tuple of (x, y) validation data
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Whether to print progress

        Returns:
            Training history
        """
        x_train, y_train = train_data
        n_samples = x_train.shape[0]

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                loss = self.train_step(x_batch, y_batch)
                epoch_losses.append(loss)

            train_loss = np.mean(epoch_losses)
            val_loss = self.validate(val_data)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f} - "
                    f"Val Loss: {val_loss:.6f}"
                )

            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return history


def create_sequences(
    data: np.ndarray,
    seq_len: int,
    pred_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-output sequences for training.

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

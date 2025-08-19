"""
Training script for C-Mamba channel correlation model.

This script provides a complete training pipeline including:
- Data loading from Bybit or Yahoo Finance
- Model configuration and initialization
- Training loop with early stopping
- Model checkpointing and evaluation
- Logging and metrics tracking

Usage:
    python train.py --source bybit --epochs 100 --seq_len 60 --pred_len 5
    python train.py --source yahoo --symbols AAPL,MSFT,GOOGL --epochs 50
"""

import argparse
import logging
import os
import json
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

from cmamba_model import CMamba, CMambaConfig, CMambaTrainer
from data_loader import MultiAssetDataLoader, CRYPTO_UNIVERSE, STOCK_UNIVERSE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train C-Mamba model for multivariate time series forecasting"
    )

    # Data arguments
    parser.add_argument(
        "--source",
        type=str,
        default="bybit",
        choices=["bybit", "yahoo"],
        help="Data source: 'bybit' for crypto, 'yahoo' for stocks"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (e.g., 'BTCUSDT,ETHUSDT' or 'AAPL,MSFT')"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="D",
        help="Data interval: 'D' (daily), 'W' (weekly)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days of historical data"
    )

    # Model arguments
    parser.add_argument(
        "--seq_len",
        type=int,
        default=60,
        help="Input sequence length"
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=5,
        help="Prediction horizon length"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=64,
        help="Model dimension"
    )
    parser.add_argument(
        "--d_state",
        type=int,
        default=16,
        help="SSM state dimension"
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of M-Mamba layers"
    )
    parser.add_argument(
        "--patch_len",
        type=int,
        default=12,
        help="Patch length for embedding"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test set ratio"
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory for saving model checkpoints"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for this experiment (used in checkpoint naming)"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def load_data(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load and prepare data for training.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (train_data, val_data, test_data, n_channels)
    """
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]

    logger.info(f"Loading data from {args.source}...")
    logger.info(f"Symbols: {symbols or ('default ' + args.source + ' universe')}")

    # Create data loader
    loader = MultiAssetDataLoader(
        symbols=symbols,
        source=args.source,
        interval=args.interval
    )

    # Load and split data
    train_data, val_data, test_data = loader.load_and_split(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        normalize=True
    )

    n_channels = train_data.shape[1]
    logger.info(f"Data loaded: {n_channels} channels")
    logger.info(f"Train samples: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return train_data, val_data, test_data, n_channels


def create_model(args: argparse.Namespace, n_channels: int) -> CMamba:
    """
    Create and initialize C-Mamba model.

    Args:
        args: Command line arguments
        n_channels: Number of input channels

    Returns:
        Initialized CMamba model
    """
    config = CMambaConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        n_channels=n_channels,
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        patch_len=args.patch_len,
        dropout=args.dropout
    )

    logger.info("Model configuration:")
    logger.info(f"  Sequence length: {config.seq_len}")
    logger.info(f"  Prediction length: {config.pred_len}")
    logger.info(f"  Channels: {config.n_channels}")
    logger.info(f"  Model dimension: {config.d_model}")
    logger.info(f"  SSM state dimension: {config.d_state}")
    logger.info(f"  Layers: {config.n_layers}")
    logger.info(f"  Patch length: {config.patch_len}")

    model = CMamba(config)
    return model


def prepare_sequences(
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
        Tuple of (X, Y) arrays
    """
    n_samples = len(data) - seq_len - pred_len + 1

    if n_samples <= 0:
        raise ValueError(
            f"Not enough data: have {len(data)} points, "
            f"need at least {seq_len + pred_len}"
        )

    X = np.zeros((n_samples, seq_len, data.shape[1]))
    Y = np.zeros((n_samples, pred_len, data.shape[1]))

    for i in range(n_samples):
        X[i] = data[i:i + seq_len]
        Y[i] = data[i + seq_len:i + seq_len + pred_len]

    return X, Y


def train(
    model: CMamba,
    train_data: np.ndarray,
    val_data: np.ndarray,
    args: argparse.Namespace
) -> Dict[str, list]:
    """
    Train the model.

    Args:
        model: C-Mamba model
        train_data: Training data
        val_data: Validation data
        args: Command line arguments

    Returns:
        Dictionary of training history
    """
    # Prepare sequences
    X_train, Y_train = prepare_sequences(train_data, args.seq_len, args.pred_len)
    X_val, Y_val = prepare_sequences(val_data, args.seq_len, args.pred_len)

    logger.info(f"Training sequences: {X_train.shape}")
    logger.info(f"Validation sequences: {X_val.shape}")

    # Create trainer
    trainer = CMambaTrainer(
        model=model,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Experiment name
    if args.experiment_name is None:
        args.experiment_name = f"cmamba_{args.source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Starting training: {args.experiment_name}")
    logger.info(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")

    start_time = time.time()

    # Train using the trainer's built-in loop
    history = trainer.train(
        train_data=(X_train, Y_train),
        val_data=(X_val, Y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")

    # Save final model
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else float("inf")
    final_epoch = len(history["train_loss"])
    save_checkpoint(
        model, args, final_epoch, final_val_loss,
        os.path.join(args.output_dir, f"{args.experiment_name}_final.json")
    )

    # Save training history
    history_path = os.path.join(args.output_dir, f"{args.experiment_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    return history


def save_checkpoint(
    model: CMamba,
    args: argparse.Namespace,
    epoch: int,
    val_loss: float,
    path: str
):
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,
        "config": {
            "seq_len": model.config.seq_len,
            "pred_len": model.config.pred_len,
            "n_channels": model.config.n_channels,
            "d_model": model.config.d_model,
            "d_state": model.config.d_state,
            "n_layers": model.config.n_layers,
            "patch_len": model.config.patch_len,
            "dropout": model.config.dropout,
        },
        "training_args": {
            "source": args.source,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "timestamp": datetime.now().isoformat()
    }

    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    logger.info(f"Checkpoint saved: {path}")


def evaluate(
    model: CMamba,
    test_data: np.ndarray,
    args: argparse.Namespace
) -> Dict[str, float]:
    """
    Evaluate model on test data.

    Args:
        model: Trained model
        test_data: Test data
        args: Command line arguments

    Returns:
        Dictionary of evaluation metrics
    """
    X_test, Y_test = prepare_sequences(test_data, args.seq_len, args.pred_len)

    logger.info(f"Evaluating on {len(X_test)} test sequences...")

    # Get predictions
    predictions = []
    for i in range(len(X_test)):
        x = X_test[i:i + 1]
        output = model.forward(x)
        predictions.append(output.predictions)

    predictions = np.concatenate(predictions, axis=0)

    # Calculate metrics
    mse = np.mean((predictions - Y_test) ** 2)
    mae = np.mean(np.abs(predictions - Y_test))
    rmse = np.sqrt(mse)

    # Per-channel metrics
    channel_mse = np.mean((predictions - Y_test) ** 2, axis=(0, 1))
    channel_mae = np.mean(np.abs(predictions - Y_test), axis=(0, 1))

    metrics = {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "channel_mse": channel_mse.tolist(),
        "channel_mae": channel_mae.tolist()
    }

    logger.info("Test Metrics:")
    logger.info(f"  MSE:  {mse:.6f}")
    logger.info(f"  MAE:  {mae:.6f}")
    logger.info(f"  RMSE: {rmse:.6f}")

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("C-Mamba Training Script")
    logger.info("=" * 60)

    # Load data
    train_data, val_data, test_data, n_channels = load_data(args)

    # Create model
    model = create_model(args, n_channels)

    # Train
    history = train(model, train_data, val_data, args)

    # Evaluate on test set
    if len(test_data) >= args.seq_len + args.pred_len:
        metrics = evaluate(model, test_data, args)

        # Save metrics
        metrics_path = os.path.join(args.output_dir, f"{args.experiment_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Test metrics saved to {metrics_path}")
    else:
        logger.warning("Test data too small for evaluation")

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

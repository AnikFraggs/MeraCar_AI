"""Reusable PyTorch building blocks: datasets, an LSTM regressor, and a train loop."""
from __future__ import annotations

import copy
import logging

from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Import the logger setup from utils.py
from .utils import setup_logger

logger = setup_logger()

class SequenceDataset(Dataset):
    """Sliding-window dataset turning a (T, F) array into (window, F) -> target samples.

    Args:
        features: 2D array of shape (T, F_features)
        targets: 2D array of shape (T, F_targets)
        window: Number of past timesteps to use as input.
        horizon: Number of future timesteps to predict. If 1, predicts the immediate 
                 next step. If >1, predicts a sequence of future steps.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, window: int, horizon: int = 1):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.window = window
        self.horizon = horizon

    def __len__(self) -> int:
        # Ensure we don't index out of bounds when fetching the horizon
        return len(self.features) - self.window - self.horizon + 1

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + self.window]
        y = self.targets[idx + self.window : idx + self.window + self.horizon].reshape(-1)
        return torch.from_numpy(x), torch.from_numpy(y)

class LSTMRegressor(nn.Module):
    """Generic multi-output LSTM regressor for time-series control/prediction."""

    def __init__(self, n_features: int, n_outputs: int, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]        
        return self.head(last)


class MLPRegressor(nn.Module):
    """Plain feed-forward regressor for tabular physics features."""

    def __init__(self, n_features: int, n_outputs: int, hidden=(128, 64)):
        super().__init__()
        dims = [n_features, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_regression(model, train_loader, val_loader, *, epochs=30, lr=1e-3,
                     device=None, weight_decay=1e-5, verbose=True, clip_grad=1.0, patience=5):
    """Train a regression model with Adam + MSE. Includes Early Stopping and Gradient Clipping.
    
    Args:
        patience: Number of epochs to wait for val_loss improvement before stopping.
        clip_grad: Max norm for gradient clipping (prevents LSTM exploding gradients).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history = {"train": [], "val": []}
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0
    
    logger.info(f"Starting training on device: {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            
            # If horizon=1, yb shape might be (batch, 1, features). Squeeze to match pred if needed.
            if yb.ndim == 3 and yb.size(1) == 1 and pred.ndim == 2:
                yb = yb.squeeze(1)
                
            loss = loss_fn(pred, yb)
            loss.backward()
            
            # Gradient Clipping
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
            opt.step()
            tr_loss += loss.item() * len(xb)
            
        tr_loss /= len(train_loader.dataset)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        history["train"].append(tr_loss)
        history["val"].append(val_loss)
        
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == 1):
            logger.info(f"  epoch {epoch:3d}/{epochs}  train={tr_loss:.4f}  val={val_loss:.4f}")
            
        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch}. Restoring best weights (val_loss={best_val_loss:.4f}).")
                break
                
    # Restore best weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        
    return model, history


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            if yb.ndim == 3 and yb.size(1) == 1 and pred.ndim == 2:
                yb = yb.squeeze(1)
            total += loss_fn(pred, yb).item() * len(xb)
    return total / len(loader.dataset)


def make_loaders(X, y, *, window=None, horizon=1, batch_size=64, val_frac=0.2):
    """Build train/val DataLoaders. If window is set, use a SequenceDataset."""
    n = len(X)
    split = int(n * (1 - val_frac))
    if window:
        train_ds = SequenceDataset(X[:split], y[:split], window, horizon)
        val_ds = SequenceDataset(X[split:], y[split:], window, horizon)
    else:
        train_ds = TensorDataset(
            torch.tensor(X[:split], dtype=torch.float32),
            torch.tensor(y[:split], dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X[split:], dtype=torch.float32),
            torch.tensor(y[split:], dtype=torch.float32),
        )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=not window),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def save_checkpoint(model, path: Path, meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta or {}}, path)
    logger.info(f"Model checkpoint saved to {path}")


def load_checkpoint(model, path: Path, device=None):
    device = device or torch.device("cpu")
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Model checkpoint loaded from {path}")
    return model, ckpt.get("meta", {})
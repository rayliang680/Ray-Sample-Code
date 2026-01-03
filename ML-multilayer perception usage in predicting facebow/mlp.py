# mlp_angle_regression.py
# A complete Multi-Layer Perceptron (MLP) regression example to predict a single angle value (continuous).
# Uses synthetic (generated) data; no real dataset required.

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Reproducibility utilities
# -------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Synthetic data generation
# -------------------------
def generate_synthetic_angle_data(
    n_samples: int = 8000,
    n_features: int = 10,
    noise_std_deg: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic features X and an angle label y (in degrees).
    The label is produced by a hidden nonlinear function + noise.
    """
    # Features: standard normal
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Hidden rule:
    # Create a nonlinear scalar signal, then map to an angle in degrees.
    # This is just for demonstration.
    w1 = np.linspace(0.2, 1.0, n_features).astype(np.float32)
    w2 = np.linspace(1.0, 0.2, n_features).astype(np.float32)

    s1 = X @ w1
    s2 = np.tanh(X @ w2)

    # Combine signals, map into [-pi, pi] then to degrees [0, 360)
    # You can change the mapping depending on how you define "angle".
    rad = np.arctan2(s2, s1)  # range [-pi, pi]
    deg = (rad * 180.0 / np.pi)  # [-180, 180]

    # Add noise in degrees
    deg_noisy = deg + np.random.randn(n_samples).astype(np.float32) * noise_std_deg

    # Option A: Predict angle directly in degrees (can be negative).
    # Option B: Wrap to [0, 360). Choose one. Here we keep it in [-180, 180] to keep regression simple.
    y = deg_noisy.astype(np.float32).reshape(-1, 1)

    return X, y


# -------------------------
# Simple standard scaler
# -------------------------
@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# -------------------------
# Dataset
# -------------------------
class AngleDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# -------------------------
# MLP model
# -------------------------
class MLPRegressor(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1)  # single continuous value: angle (deg)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# Train / eval loops
# -------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * Xb.size(0)
        n += Xb.size(0)

    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Returns:
      - MSE loss
      - MAE in degrees
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n = 0

    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        pred = model(Xb)
        mse = loss_fn(pred, yb)
        mae = (pred - yb).abs().mean()

        total_mse += mse.item() * Xb.size(0)
        total_mae += mae.item() * Xb.size(0)
        n += Xb.size(0)

    return total_mse / max(n, 1), total_mae / max(n, 1)


# -------------------------
# Main
# -------------------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Generate synthetic data
    n_samples = 8000
    n_features = 10
    X, y = generate_synthetic_angle_data(
        n_samples=n_samples,
        n_features=n_features,
        noise_std_deg=2.0,
    )

    # 2) Train/val/test split
    idx = np.random.permutation(n_samples)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # 3) Standardize features (fit on train only)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 4) DataLoaders
    batch_size = 128
    train_loader = DataLoader(AngleDataset(X_train_s, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AngleDataset(X_val_s, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(AngleDataset(X_test_s, y_test), batch_size=batch_size, shuffle=False)

    # 5) Model, loss, optimizer
    model = MLPRegressor(n_features=n_features).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 6) Training loop with simple early stopping
    best_val_mse = float("inf")
    best_state = None
    patience = 15
    wait = 0
    epochs = 200

    for epoch in range(1, epochs + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_mse, val_mae = evaluate(model, val_loader, loss_fn, device)

        if val_mse < best_val_mse - 1e-6:
            best_val_mse = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train MSE={train_mse:.4f} | val MSE={val_mse:.4f} | val MAE={val_mae:.3f} deg")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}. Best val MSE={best_val_mse:.4f}")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # 7) Final evaluation on test set
    test_mse, test_mae = evaluate(model, test_loader, loss_fn, device)
    print(f"\nTest MSE={test_mse:.4f} | Test MAE={test_mae:.3f} deg")

    # 8) Inference: predict angle for new samples
    # Example: a single random sample
    x_new = np.random.randn(1, n_features).astype(np.float32)
    x_new_s = scaler.transform(x_new)
    x_new_t = torch.from_numpy(x_new_s).float().to(device)

    with torch.no_grad():
        pred_deg = model(x_new_t).cpu().numpy().reshape(-1)[0]

    print(f"\nPredicted angle (deg): {pred_deg:.3f}")

    # If you need angle normalization:
    # -180..180 -> 0..360:
    pred_0_360 = (pred_deg + 360.0) % 360.0
    print(f"Predicted angle normalized to [0, 360): {pred_0_360:.3f}")


if __name__ == "__main__":
    main()

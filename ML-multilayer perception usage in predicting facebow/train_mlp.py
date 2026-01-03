# train_mlp.py
# Load synthetic data from .npz and train an MLP regressor to predict angle (degrees).

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NumpyIndexDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray):
        self.X = torch.from_numpy(X[indices]).float()
        self.y = torch.from_numpy(y[indices]).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MLPRegressor(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total = 0.0
    n = 0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()

        total += loss.item() * Xb.size(0)
        n += Xb.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> tuple[float, float]:
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


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1) Load from file
    data_path = "angle_data.npz"
    d = np.load(data_path, allow_pickle=True)

    X = d["X"].astype(np.float32)         # standardized features
    y = d["y"].astype(np.float32)         # degrees
    train_idx = d["train_idx"]
    val_idx = d["val_idx"]
    test_idx = d["test_idx"]

    n_features = X.shape[1]
    print(f"Loaded: X {X.shape}, y {y.shape}")

    # 2) DataLoaders
    batch_size = 128
    train_loader = DataLoader(NumpyIndexDataset(X, y, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NumpyIndexDataset(X, y, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(NumpyIndexDataset(X, y, test_idx), batch_size=batch_size, shuffle=False)

    # 3) Model
    model = MLPRegressor(n_features=n_features).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 4) Train + early stopping
    best_val_mse = float("inf")
    best_state = None
    patience = 15
    wait = 0
    epochs = 300

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

    if best_state is not None:
        model.load_state_dict(best_state)

    # 5) Test
    test_mse, test_mae = evaluate(model, test_loader, loss_fn, device)
    print(f"\nTest MSE={test_mse:.4f} | Test MAE={test_mae:.3f} deg")

    # 6) Inference demo (new random sample)
    # Note: X in file is standardized using train mean/std. If you want to standardize new input the same way,
    # use d["mean"], d["std"] saved in the npz.
    mean = d["mean"].astype(np.float32)
    std = d["std"].astype(np.float32)

    x_new = np.random.randn(1, n_features).astype(np.float32)
    x_new_s = (x_new - mean) / std
    x_new_t = torch.from_numpy(x_new_s).float().to(device)

    with torch.no_grad():
        pred_deg = model(x_new_t).cpu().numpy().reshape(-1)[0]

    pred_0_360 = (pred_deg + 360.0) % 360.0
    print(f"\nPredicted angle (deg): {pred_deg:.3f}")
    print(f"Predicted angle normalized to [0, 360): {pred_0_360:.3f}")


if __name__ == "__main__":
    main()

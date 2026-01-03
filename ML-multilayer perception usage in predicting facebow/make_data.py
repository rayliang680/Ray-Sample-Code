# make_data.py
# Generate synthetic angle regression data and save to an .npz file.

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def generate_synthetic_angle_data(
    n_samples: int = 8000,
    n_features: int = 10,
    noise_std_deg: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic features X and an angle label y (in degrees).
    y is a continuous value in approximately [-180, 180] with noise.
    """
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    w1 = np.linspace(0.2, 1.0, n_features).astype(np.float32)
    w2 = np.linspace(1.0, 0.2, n_features).astype(np.float32)

    s1 = X @ w1
    s2 = np.tanh(X @ w2)

    rad = np.arctan2(s2, s1)             # [-pi, pi]
    deg = rad * 180.0 / np.pi            # [-180, 180]
    deg_noisy = deg + np.random.randn(n_samples).astype(np.float32) * noise_std_deg

    y = deg_noisy.astype(np.float32).reshape(-1, 1)
    return X, y


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def main():
    set_seed(42)

    # Config
    out_path = "angle_data.npz"
    n_samples = 8000
    n_features = 10
    noise_std_deg = 2.0

    # Generate
    X, y = generate_synthetic_angle_data(
        n_samples=n_samples,
        n_features=n_features,
        noise_std_deg=noise_std_deg,
    )

    # Split indices: train/val/test = 70/15/15
    idx = np.random.permutation(n_samples)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    # Fit scaler ONLY on train, then transform all splits
    mean, std = standardize_fit(X[train_idx])
    Xs = standardize_apply(X, mean, std)

    # Save everything needed for training to one file
    np.savez_compressed(
        out_path,
        X=Xs,           # standardized features
        y=y,            # labels in degrees
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        mean=mean,      # for future inference if you want to standardize new data similarly
        std=std,
        meta=np.array(
            [n_samples, n_features, noise_std_deg],
            dtype=np.float32
        ),
    )

    print(f"Saved dataset to: {out_path}")
    print(f"X shape: {Xs.shape}, y shape: {y.shape}")
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")


if __name__ == "__main__":
    main()

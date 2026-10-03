from __future__ import annotations
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10

def load_split(processed_dir: str = "data/processed"):
    train_path = os.path.join(processed_dir, "train.npz")
    eval_path = os.path.join(processed_dir, "eval.npz")
    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        raise FileNotFoundError(f"Không tìm thấy file npz tại {processed_dir}. Hãy chạy scripts/split_data.py trước!")
    train_data = np.load(train_path)
    eval_data = np.load(eval_path)
    X_train_full = train_data["X"].astype(np.float32)
    y_train_full = train_data["y"].astype(np.int64)
    X_eval = eval_data["X"].astype(np.float32)
    y_eval = eval_data["y"].astype(np.int64)
    eval_row_id = eval_data["row_id"].astype(np.int64)
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id

def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_fraction, random_state=seed, stratify=y)
    return X_tr, y_tr, X_val, y_val

def fit_standardizer(X_tr):
    mean = np.mean(X_tr[:, :N_NUMERIC], axis=0).astype(np.float32)
    std = np.std(X_tr[:, :N_NUMERIC], axis=0).astype(np.float32)
    std = np.where(std < 1e-7, 1.0, std).astype(np.float32)
    return mean, std

def apply_standardizer(X, mean, std):
    X_out = X.copy()
    X_out[:, :N_NUMERIC] = (X_out[:, :N_NUMERIC] - mean) / std
    return X_out

def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    dev = torch.device(device)
    data = {
        "X_tr": torch.as_tensor(X_tr, dtype=torch.float32, device=dev),
        "y_tr": torch.as_tensor(y_tr, dtype=torch.int64, device=dev),
        "X_val": torch.as_tensor(X_val, dtype=torch.float32, device=dev),
        "y_val": torch.as_tensor(y_val, dtype=torch.int64, device=dev),
        "X_eval": torch.as_tensor(X_eval, dtype=torch.float32, device=dev),
        "y_eval": torch.as_tensor(y_eval, dtype=torch.int64, device=dev),
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }
    n_tr, n_val, n_eval = len(data["X_tr"]), len(data["X_val"]), len(data["X_eval"])
    print(f"Dữ liệu sẵn sàng trên {device}: Train={n_tr:,}, Val={n_val:,}, Eval={n_eval:,}")
    classes, counts = np.unique(y_val, return_counts=True)
    majority_class = classes[np.argmax(counts)]
    majority_acc = np.mean(y_val == majority_class)
    print(f"Đoán đa số (lớp {majority_class}) trên val: accuracy = {majority_acc:.4f} (≈ 0.4876)")
    return data

def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    n = len(X)
    perm = torch.randperm(n, generator=generator, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]

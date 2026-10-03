from __future__ import annotations
import copy, os, random, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce", optimizer="sgd_momentum", lr=0.05,
    weight_decay=0.0, momentum=0.9, batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None, precision="fp32", seed=1,
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def macro_f1_from_confusion(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())

@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        preds.append(model(xb).argmax(dim=1))
    return torch.cat(preds, dim=0)

def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return F.mse_loss(logits, y_onehot)
    raise ValueError(f"Hàm mất mát không hợp lệ: {loss_name}")

@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    model.eval()
    total_loss, total_correct = 0.0, 0
    cm = np.zeros((7, 7), dtype=np.int64)
    n = len(X)
    for i in range(0, n, batch_size):
        xb, yb = X[i:i + batch_size], y[i:i + batch_size]
        logits = model(xb)
        total_loss += float(compute_loss(logits, yb, loss_name).item()) * len(xb)
        pred = logits.argmax(dim=1)
        total_correct += int((pred == yb).sum().item())
        np.add.at(cm, (yb.cpu().numpy(), pred.cpu().numpy()), 1)
    return {"loss": float(total_loss / n), "acc": float(total_correct / n), "macro_f1": float(macro_f1_from_confusion(cm))}

def run_experiment(cfg: dict, data: dict) -> dict:
    set_seed(cfg["seed"])
    device = data["X_tr"].device
    is_cuda = (device.type == "cuda")
    if is_cuda: torch.cuda.reset_peak_memory_stats(device)

    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=float(cfg.get("dropout", 0.0)), init=str(cfg.get("init", "he"))).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden]

    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg.get("weight_decay", 0.0), momentum=cfg.get("momentum", 0.9))

    precision = cfg.get("precision", "fp32")
    scaler = torch.amp.GradScaler(device.type) if (precision == "fp16" and is_cuda) else None
    amp_dtype = torch.float16 if precision == "fp16" else (torch.bfloat16 if precision == "bf16" else torch.float32)

    step0_loss = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])["loss"]
    n_train_sub = min(50000, len(data["X_tr"]))
    X_tr_eval, y_tr_eval = data["X_tr"][:n_train_sub], data["y_tr"][:n_train_sub]

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []}
    best_val_loss, best_epoch, best_state = float("inf"), 1, None
    best_val_acc, best_val_macro_f1, diverged = 0.0, 0.0, False

    for epoch in range(1, int(cfg["epochs"]) + 1):
        if is_cuda: torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()
        epoch_grad_norms = []
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], int(cfg["batch"]), shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            if precision in ("fp16", "bf16") and is_cuda:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = compute_loss(model(xb), yb, cfg["loss"])
            else:
                loss = compute_loss(model(xb), yb, cfg["loss"])

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True; break

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                epoch_grad_norms.append(clip_gradients(model.parameters(), cfg.get("clip_norm")))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                epoch_grad_norms.append(clip_gradients(model.parameters(), cfg.get("clip_norm")))
                optimizer.step()

        if is_cuda: torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0
        if diverged: break

        tr_res = evaluate(model, X_tr_eval, y_tr_eval, loss_name=cfg["loss"])
        val_res = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
        mean_gn = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_res["loss"])
        history["val_loss"].append(val_res["loss"])
        history["val_acc"].append(val_res["acc"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(epoch_time)

        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            best_epoch = epoch
            best_val_acc = val_res["acc"]
            best_val_macro_f1 = val_res["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if is_cuda else 0.0
    summary = {
        "step0_loss": float(step0_loss), "best_val_loss": float(best_val_loss), "best_epoch": int(best_epoch),
        "final_train_loss": float(history["train_loss"][-1] if history["train_loss"] else float("nan")),
        "final_val_loss": float(history["val_loss"][-1] if history["val_loss"] else float("nan")),
        "val_acc": float(best_val_acc), "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"]) if history["epoch_time_s"] else 0.0),
        "peak_mem_MB": float(peak_mem_mb), "diverged": diverged,
    }
    if best_state is None and not diverged:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}

def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame({"row_id": row_id.astype(int), "pred": preds.astype(int)}).to_csv(path, index=False)
    print(f"Đã lưu dự đoán ({len(row_id):,} dòng) vào: {path}")

def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    model = MLP(hidden=tuple(cfg["hidden"]), dropout=float(cfg.get("dropout", 0.0)), init=str(cfg.get("init", "he"))).to(data["X_eval"].device)
    model.load_state_dict(result["best_state"])
    model.eval()
    write_predictions(data["eval_row_id"], predict(model, data["X_eval"]).cpu().numpy(), pred_path)

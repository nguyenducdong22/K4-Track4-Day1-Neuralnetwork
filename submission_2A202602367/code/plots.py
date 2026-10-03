from __future__ import annotations
import os, matplotlib.pyplot as plt

def plot_run(result: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cfg, hist, summary = result["cfg"], result["history"], result.get("summary", {})
    epochs = hist.get("epoch", [])
    if not epochs: return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    best_ep = summary.get("best_epoch", None)

    # 1. Loss
    axes[0].plot(epochs, hist["train_loss"], label="Train (eval mode)", color="#1f77b4", marker="o", markersize=3)
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", marker="s", markersize=3)
    if best_ep in epochs: axes[0].axvline(best_ep, color="green", linestyle="--", label=f"Best ep ({best_ep})")
    axes[0].set_title("Loss theo Epoch"); axes[0].set_xlabel("Epoch"); axes[0].grid(True, linestyle=":", alpha=0.6); axes[0].legend()

    # 2. Metrics
    axes[1].plot(epochs, hist["val_acc"], label="Val Acc", color="#2ca02c", marker="^", markersize=3)
    if "val_macro_f1" in hist: axes[1].plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#d62728", marker="d", markersize=3)
    if best_ep in epochs: axes[1].axvline(best_ep, color="green", linestyle="--", label=f"Best ep ({best_ep})")
    axes[1].set_title("Metrics trên Val"); axes[1].set_xlabel("Epoch"); axes[1].grid(True, linestyle=":", alpha=0.6); axes[1].legend()

    # 3. Grad Norm
    axes[2].plot(epochs, hist["grad_norm"], label="Grad Norm (L2)", color="#9467bd", marker="x", markersize=4)
    if cfg.get("clip_norm") is not None: axes[2].axhline(cfg["clip_norm"], color="red", linestyle=":", label=f"Clip c={cfg['clip_norm']}")
    axes[2].set_title("Grad Norm (trước clip)"); axes[2].set_xlabel("Epoch"); axes[2].grid(True, linestyle=":", alpha=0.6); axes[2].legend()

    fig.suptitle(f"Exp: {cfg.get('exp_id')} | Opt: {cfg.get('optimizer')} (lr={cfg.get('lr')})", fontsize=12, fontweight="bold")
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for res in results:
        cfg, hist = res["cfg"], res["history"]
        if metric in hist and hist[metric]:
            ax.plot(hist.get("epoch", list(range(1, len(hist[metric]) + 1))), hist[metric], marker="o", markersize=3, label=cfg.get("exp_id"))
    ax.set_title(title or f"So sánh {metric}"); ax.set_xlabel("Epoch"); ax.set_ylabel(metric); ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

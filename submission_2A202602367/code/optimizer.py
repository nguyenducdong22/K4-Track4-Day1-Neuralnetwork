from __future__ import annotations
import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")

def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    if name not in OPTIMIZERS:
        raise ValueError(f"Bộ tối ưu {name} không hợp lệ.")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    if name is None: return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    raise ValueError(f"Scheduler không hỗ trợ: {name}")

def clip_gradients(params, max_norm: float | None) -> float:
    param_list = [p for p in params if p.grad is not None]
    if not param_list: return 0.0
    if max_norm is None:
        return float(torch.nn.utils.clip_grad_norm_(param_list, float("inf")))
    return float(torch.nn.utils.clip_grad_norm_(param_list, max_norm))

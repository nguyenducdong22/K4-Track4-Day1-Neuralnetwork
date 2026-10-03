from __future__ import annotations
import torch
import torch.nn as nn

EXPECTED_PARAMS = {
    (256, 128): 47_879,
    (512, 256): 161_287,
    (256, 128, 64): 55_687,
}

class MLP(nn.Module):
    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        self.hidden = tuple(hidden)
        layers = []
        in_dim = in_features
        for h in hidden:
            layers.append(nn.Linear(in_dim, h, bias=True))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes, bias=True))
        self.net = nn.Sequential(*layers)
        init_weights(self, init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def init_weights(model: nn.Module, init: str) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight)
            elif init == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif init == "zeros":
                nn.init.zeros_(m.weight)
            elif init == "default":
                pass
            else:
                raise ValueError(f"Khởi tạo không hỗ trợ: {init}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    model.eval()
    stds = []
    h = x
    for layer in model.net:
        h = layer(h)
        if isinstance(layer, nn.Linear):
            stds.append(float(h.std().item()))
    return stds

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseDropoutBlock(nn.Module):
    """
    Pre-activation block:
        LN(in_dim) -> GELU -> Linear(in_dim -> out_dim) -> Dropout
    """
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.net(x)

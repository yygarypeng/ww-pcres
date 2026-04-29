import torch.nn as nn
from layers import _AttnFFN

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

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.3):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        
        self.ffn = _AttnFFN(d_model, d_model * 4, dropout)
    
    def forward(self, queries, context, key_padding_mask=None):
        res = queries
        q = self.norm_q(queries)
        kv = self.norm_kv(context)
        
        attn_out, _ = self.mha(query=q, key=kv, value=kv, key_padding_mask=key_padding_mask)
        x = res + self.dropout(attn_out)
        
        x = x + self.dropout(self.ffn(x))
        return x

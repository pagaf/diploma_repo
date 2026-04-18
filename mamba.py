import torch
import torch.nn as nn
from mamba_ssm import Mamba

class BiMamba2Block(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.mamba_fwd = Mamba(
            d_model=hidden_dim,
            d_state=64,
            d_conv=4,
            expand=2,
            dt_rank="auto"
        )
        self.mamba_bwd = Mamba(
            d_model=hidden_dim,
            d_state=64,
            d_conv=4,
            expand=2,
            dt_rank="auto"
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        h = self.ln(x)
        h_fwd = self.mamba_fwd(h)
        h_bwd = self.mamba_bwd(h.flip(dims=[1])).flip(dims=[1])
        h_concat = torch.cat([h_fwd, h_bwd], dim=-1)
        out = self.proj(h_concat)
        return x + out

class TSPMamba(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        self.edge_emb = nn.Linear(17, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList([BiMamba2Block(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, edges_feat, xt, t):
        B, E = xt.shape
        h = self.edge_emb(edges_feat)
        t_emb = self.time_mlp(t.unsqueeze(-1))
        h = h + t_emb

        # Mamba needs [B, L, D]
        h = h.unsqueeze(0) 

        for layer in self.layers:
            h = layer(h)

        h = h.squeeze(0)
        out = self.head(h).squeeze(-1)
        return out

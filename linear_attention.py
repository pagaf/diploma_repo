import torch
import torch.nn as nn

class LinearAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x):
        L, D = x.shape
        h = self.ln1(x)

        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(L, self.num_heads, D // self.num_heads)
        k = k.view(L, self.num_heads, D // self.num_heads)
        v = v.view(L, self.num_heads, D // self.num_heads)

        # Performer/Linear Attention style (elu + 1 kernel)
        q = torch.nn.functional.elu(q) + 1.0
        k = torch.nn.functional.elu(k) + 1.0

        kv = torch.einsum('lhd,lhm->hdm', k, v)
        z = 1.0 / (torch.einsum('lhd,hd->lh', q, k.sum(dim=0)) + 1e-6)

        out = torch.einsum('lhd,hdm,lh->lhm', q, kv, z)
        out = out.contiguous().view(L, D)
        out = self.proj(out)

        x = x + out
        x = x + self.mlp(self.ln2(x))
        return x

class TSPLinearAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        self.edge_emb = nn.Linear(17, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList([LinearAttentionBlock(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, edges_feat, xt, t):
        h = self.edge_emb(edges_feat)
        t_emb = self.time_mlp(t.unsqueeze(-1))
        h = h + t_emb

        for layer in self.layers:
            h = layer(h)

        out = self.head(h).squeeze(-1)
        return out

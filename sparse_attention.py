import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, window_size=50):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
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
        q = q.view(L, self.num_heads, D // self.num_heads).transpose(0, 1)
        k = k.view(L, self.num_heads, D // self.num_heads).transpose(0, 1)
        v = v.view(L, self.num_heads, D // self.num_heads).transpose(0, 1)

        # Sliding window attention mask
        idx = torch.arange(L, device=x.device)
        dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        mask = (dist > self.window_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        scores.masked_fill_(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(0, 1).contiguous().view(L, D)
        out = self.proj(out)

        x = x + out
        x = x + self.mlp(self.ln2(x))
        return x

class TSPSparseAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        self.edge_emb = nn.Linear(17, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers = nn.ModuleList([SparseAttentionBlock(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, edges_feat, xt, t):
        h = self.edge_emb(edges_feat)
        t_emb = self.time_mlp(t.unsqueeze(-1))
        h = h + t_emb

        for layer in self.layers:
            h = layer(h)

        out = self.head(h).squeeze(-1)
        return out

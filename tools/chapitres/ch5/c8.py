# CELLULE 8 — Un petit encodeur Transformer en PyTorch (CPU), avec quatre options de position.
import math, time
import torch
from torch import nn
import torch.nn.functional as F
torch.set_num_threads(4)

def sinusoidal(n, d):
    pos = torch.arange(n)[:, None]
    freq = torch.exp(-math.log(10000.) * torch.arange(0, d, 2) / d)
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return pe

def rope(x):
    """Rotation des paires de coordonnées : x de forme (B, h, n, d_h)."""
    n, dh = x.shape[-2], x.shape[-1]
    theta = torch.exp(-math.log(10000.) * torch.arange(0, dh, 2) / dh)
    ang = torch.arange(n)[:, None] * theta            # (n, d_h/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

class MHA(nn.Module):
    def __init__(self, d, h, use_rope=False):
        super().__init__()
        self.h, self.use_rope = h, use_rope
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
    def forward(self, x):
        B, n, d = x.shape
        q, k, v = self.qkv(x).view(B, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q, k = rope(q), rope(k)
        att = (q @ k.transpose(-1, -2) / math.sqrt(d // self.h)).softmax(-1)
        self.last_attention = att.detach()
        return self.out((att @ v).transpose(1, 2).reshape(B, n, d))

class Block(nn.Module):
    def __init__(self, d, h, use_rope):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MHA(d, h, use_rope)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))          # pré-normalisation
        return x + self.ffn(self.ln2(x))

class Encoder(nn.Module):
    def __init__(self, vocab, n, d=64, h=4, layers=2, position='learned'):
        super().__init__()
        self.position = position
        self.emb = nn.Embedding(vocab, d)
        if position == 'learned':
            self.pos = nn.Parameter(0.02 * torch.randn(n, d))
        elif position == 'sinusoidal':
            self.register_buffer('pos', sinusoidal(n, d))
        self.blocks = nn.ModuleList(Block(d, h, position == 'rope') for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)
    def forward(self, tokens):
        x = self.emb(tokens)
        if self.position in ('learned', 'sinusoidal'):
            x = x + self.pos[:tokens.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

model = Encoder(vocab=10, n=8)
print('Paramètres :', sum(p.numel() for p in model.parameters()))

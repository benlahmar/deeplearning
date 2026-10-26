# CELLULE 4 — LoRA : W = W₀ + (α/r)·B A, avec B initialisée à zéro.
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha=None):
        super().__init__()
        self.base, self.r, self.scale = base, r, (alpha or 2 * r) / r
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T
    def merged_weight(self):
        return self.base.weight + self.scale * self.B @ self.A

def add_lora(model, r):
    for blk in model.blocks:
        for name in ('qkv', 'proj', 'fc1', 'fc2'):
            setattr(blk, name, LoRALinear(getattr(blk, name), r))
    return model

class Classifier(nn.Module):
    """Tronc (préentraîné ou non) + tête linéaire sur la moyenne des positions."""
    def __init__(self, trunk):
        super().__init__()
        self.trunk, self.head = trunk, nn.Linear(64, K)
    def forward(self, x):
        return self.head(self.trunk.features(x).mean(1))

def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

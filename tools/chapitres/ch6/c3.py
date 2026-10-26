# CELLULE 3 — Un petit décodeur causal (type GPT) et son préentraînement.
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
    def forward(self, x):
        B, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, n, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))

class TinyGPT(nn.Module):
    def __init__(self, d=64, h=4, layers=2):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Parameter(0.02 * torch.randn(T, d))
        self.blocks = nn.ModuleList(Block(d, h) for _ in range(layers))
        self.ln, self.lm_head = nn.LayerNorm(d), nn.Linear(d, V)
    def features(self, x):
        h = self.emb(x) + self.pos[:x.shape[1]]
        for b in self.blocks:
            h = b(h)
        return self.ln(h)                          # (B, n, d)
    def forward(self, x):
        return self.lm_head(self.features(x))

def lm_loss(model, X):
    with torch.no_grad():
        logits = model(X[:, :-1])
        return F.cross_entropy(logits.reshape(-1, V), X[:, 1:].reshape(-1)).item()

def pretrain(steps=2500, seed=0):
    torch.manual_seed(seed)
    model = TinyGPT()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=.1)
    gen, history = torch.Generator().manual_seed(seed), []
    for step in range(steps):
        xb = X_pre[torch.randint(0, len(X_pre), (64,), generator=gen)]
        loss = F.cross_entropy(model(xb[:, :-1]).reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 50 == 0:
            history.append((step, loss.item()))
    return model, history

t0 = time.time()
pretrained, pre_history = pretrain()
print(f'Préentraînement : {time.time()-t0:.0f} s ; perte LM validation = {lm_loss(pretrained, X_lm_val):.3f}',
      f'(uniforme : log V = {math.log(V):.3f})')

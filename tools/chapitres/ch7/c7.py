# CELLULE 7 — JEPA pour des séquences : prédire la représentation de la suite, pas ses jetons.
# On reprend le langage à quatre thèmes du chapitre 6 (même générateur).
V, T, K = 24, 32, 4
g = torch.Generator().manual_seed(0)
TRANS = (2.0 * torch.randn(V, V, generator=g) + 1.5 * torch.randn(K, V, V, generator=g)).softmax(-1)
def sample(n, seed):
    gen = torch.Generator().manual_seed(seed)
    z = torch.randint(0, K, (n,), generator=gen)
    x = torch.empty(n, T, dtype=torch.long); x[:, 0] = torch.randint(0, V, (n,), generator=gen)
    for t in range(1, T):
        x[:, t] = torch.multinomial(TRANS[z, x[:, t - 1]], 1, generator=gen).squeeze(1)
    return x, z
S_pre, _ = sample(20000, 1); S_tr, zs_tr = sample(64, 3); S_te, zs_te = sample(4000, 4)

class SeqEncoder(nn.Module):
    """Sac de transitions (x_t, x_{t+1}) plongées, puis MLP : assez pour capter un thème."""
    def __init__(self, d=64):
        super().__init__()
        self.bag = nn.EmbeddingBag(V * V, 64, mode='mean')
        self.mlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, d))
    def forward(self, x):
        return self.mlp(self.bag(x[:, :-1] * V + x[:, 1:]))

def seq_jepa(mode='ema', steps=1500, seed=0, m=.99):
    """mode : 'ema' (cible EMA sans gradient), 'naive' (cible = encodeur en ligne, avec gradient)."""
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    ctx = SeqEncoder(); tgt = copy.deepcopy(ctx)
    for p in tgt.parameters():
        p.requires_grad_(False)
    pred = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
    opt = torch.optim.Adam(list(ctx.parameters()) + list(pred.parameters()), lr=1e-3)
    for step in range(steps):
        x = S_pre[torch.randint(0, len(S_pre), (256,), generator=gen)]
        past, future = x[:, :16], x[:, 16:]                   # contexte : 16 jetons ; cible : les 16 suivants
        s_hat = pred(ctx(past))
        s_y = ctx(future) if mode == 'naive' else tgt(future).detach()
        loss = F.mse_loss(F.normalize(s_hat, dim=1), F.normalize(s_y, dim=1))
        opt.zero_grad(); loss.backward(); opt.step()
        if mode == 'ema':
            ema_update(tgt, ctx, m)
    return ctx

seq_results = {}
for name, enc in [('aléatoire', SeqEncoder()), ('JEPA naïf (sans EMA)', seq_jepa('naive')), ('JEPA (cible EMA)', seq_jepa('ema'))]:
    with torch.no_grad():
        Ztr, Zte = enc(S_tr), enc(S_te); Zall = enc(S_pre[:2000])
    seq_results[name] = (linear_probe(Ztr, zs_tr, Zte, zs_te), effective_rank(Zall))
    print(f'{name:22s} thème, sonde 64 étiquettes : {seq_results[name][0]:.3f} ; rang effectif {seq_results[name][1]:.1f}')
print('Référence chapitre 6 : modèle de langage autorégressif + sonde = 0,978')

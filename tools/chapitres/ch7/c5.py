# CELLULE 5 — Sans négatifs : prédicteur + cible à moyenne mobile (BYOL), puis une mini I-JEPA.
def ema_update(target, online, m):
    with torch.no_grad():
        for pt, po in zip(target.parameters(), online.parameters()):
            pt.mul_(m).add_(po, alpha=1 - m)

def train_byol(steps=1500, seed=0, m=.99):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    enc, proj = encoder(), projector()
    pred = nn.Sequential(nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 128))
    t_enc, t_proj = copy.deepcopy(enc), copy.deepcopy(proj)
    for p in list(t_enc.parameters()) + list(t_proj.parameters()):
        p.requires_grad_(False)
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()) + list(pred.parameters()), lr=1e-3)
    history = []
    for step in range(steps):
        x = X_ssl[torch.randint(0, len(X_ssl), (256,), generator=gen)]
        v1, v2 = augment(x, gen), augment(x, gen)
        p1, p2 = pred(proj(enc(v1))), pred(proj(enc(v2)))
        with torch.no_grad():
            t1, t2 = t_proj(t_enc(v1)), t_proj(t_enc(v2))      # cibles : pas de gradient
        loss = (2 - 2 * F.cosine_similarity(p1, t2).mean()) + (2 - 2 * F.cosine_similarity(p2, t1).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        ema_update(t_enc, enc, m); ema_update(t_proj, proj, m)
        if step % 100 == 0:
            history.append((step, effective_rank(embed(enc, X_ssl[:512]))))
    return enc, history

QUADS = [(0, 0), (0, 4), (4, 0), (4, 4)]
def block_mask(q):
    m = torch.zeros(8, 8); i, j = QUADS[q]; m[i:i + 4, j:j + 4] = 1.; return m

def train_ijepa(steps=1500, seed=0, m=.99):
    """Contexte : l'image privée d'un quart. Cible : la représentation (EMA) de ce quart seul."""
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    ctx_enc = encoder()
    tgt_enc = copy.deepcopy(ctx_enc)
    for p in tgt_enc.parameters():
        p.requires_grad_(False)
    pred = nn.Sequential(nn.Linear(64 + 4, 256), nn.ReLU(), nn.Linear(256, 64))   # conditionné par la position
    opt = torch.optim.Adam(list(ctx_enc.parameters()) + list(pred.parameters()), lr=1e-3)
    masks = torch.stack([block_mask(q) for q in range(4)])
    history = []
    for step in range(steps):
        x = X_ssl[torch.randint(0, len(X_ssl), (256,), generator=gen)]
        q = torch.randint(0, 4, (len(x),), generator=gen)
        mk = masks[q]
        s_x = ctx_enc(x * (1 - mk))                                   # contexte
        with torch.no_grad():
            s_y = tgt_enc(x * mk)                                     # cible dans l'espace latent
        s_hat = pred(torch.cat([s_x, F.one_hot(q, 4).float()], 1))
        loss = F.mse_loss(F.layer_norm(s_hat, (64,)), F.layer_norm(s_y, (64,)))
        opt.zero_grad(); loss.backward(); opt.step()
        ema_update(tgt_enc, ctx_enc, m)
        if step % 100 == 0:
            history.append((step, effective_rank(embed(ctx_enc, X_ssl[:512]))))
    return ctx_enc, history

# CELLULE 4 — Quatre objectifs sur deux vues : invariance seule, InfoNCE, VICReg, VICReg sans variance.
def info_nce(z1, z2, tau=.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.T / tau
    sim.fill_diagonal_(-float('inf'))                 # un exemple n'est pas son propre négatif
    n = z1.shape[0]
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)])
    return F.cross_entropy(sim, target)

def vicreg(z1, z2, lam=25., mu=25., nu=1., use_variance=True):
    inv = F.mse_loss(z1, z2)
    var = cov = 0.
    for z in (z1, z2):
        z = z - z.mean(0)
        if use_variance:
            var = var + F.relu(1 - torch.sqrt(z.var(0) + 1e-4)).mean()
        c = (z.T @ z) / (len(z) - 1)
        cov = cov + (c - torch.diag(torch.diag(c))).pow(2).sum() / z.shape[1]
    return lam * inv + mu * var + nu * cov

def train_joint(loss_name, steps=1500, seed=0):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    enc, proj = encoder(), projector()
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=1e-3)
    history = []
    for step in range(steps):
        x = X_ssl[torch.randint(0, len(X_ssl), (256,), generator=gen)]
        z1, z2 = proj(enc(augment(x, gen))), proj(enc(augment(x, gen)))
        if loss_name == 'invariance':
            loss = F.mse_loss(z1, z2)
        elif loss_name == 'infonce':
            loss = info_nce(z1, z2)
        else:
            loss = vicreg(z1, z2, use_variance=(loss_name == 'vicreg'))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            history.append((step, effective_rank(embed(enc, X_ssl[:512]))))
    return enc, history

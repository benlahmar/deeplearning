# CELLULE 3 — Encodeur, projecteur, et trois mesures des représentations.
def encoder():
    return nn.Sequential(nn.Flatten(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64))
def projector(d_in=64, d=128):
    return nn.Sequential(nn.Linear(d_in, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, d))

@torch.no_grad()
def embed(enc, X):
    enc.eval(); Z = enc(X); enc.train(); return Z

def linear_probe(Z_tr, y_tr, Z_te, y_te, steps=500):
    mu, sd = Z_tr.mean(0), Z_tr.std(0) + 1e-6
    head = nn.Linear(Z_tr.shape[1], 10)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        loss = F.cross_entropy(head((Z_tr - mu) / sd), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    return (head((Z_te - mu) / sd).argmax(1) == y_te).float().mean().item()

def knn(Z_tr, y_tr, Z_te, y_te, k=5):
    A, B = F.normalize(Z_tr, dim=1), F.normalize(Z_te, dim=1)
    idx = (B @ A.T).topk(k, dim=1).indices
    return (y_tr[idx].mode(1).values == y_te).float().mean().item()

def effective_rank(Z):
    s = torch.linalg.svdvals(Z - Z.mean(0))
    p = s / s.sum()
    return float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))

def evaluate(enc):
    Ztr, Zte, Zsh = embed(enc, X_ssl), embed(enc, X_test), embed(enc, X_test_shift)
    p50 = [linear_probe(Ztr[f], y_ssl[f], Zte, y_test) for f in FEW]
    return dict(probe50=float(np.mean(p50)), probe50_sd=float(np.std(p50)),
                probe_all=linear_probe(Ztr, y_ssl, Zte, y_test),
                shift_all=linear_probe(Ztr, y_ssl, Zsh, y_test),
                knn=knn(Ztr, y_ssl, Zte, y_test), rank=effective_rank(Ztr),
                spectrum=torch.linalg.svdvals(Ztr - Ztr.mean(0)).tolist())

# CELLULE 5 — Cinq stratégies, trois graines, même budget : 300 mises à jour sur 64 exemples.
def adapt(method, seed, Xs=None, zs=None, steps=300):
    Xs, zs = (X_tr, z_tr) if Xs is None else (Xs, zs)
    torch.manual_seed(seed)
    trunk = TinyGPT() if method == 'scratch' else copy.deepcopy(pretrained)
    if method == 'probe' or method.startswith('lora'):
        for p in trunk.parameters():
            p.requires_grad_(False)                 # tronc gelé
    if method.startswith('lora'):
        add_lora(trunk, int(method.split('-')[1]))
    model = Classifier(trunk)
    lr = {'scratch': 1e-3, 'probe': 1e-2, 'full': 3e-4}.get(method, 3e-3)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=.01)
    gen = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randperm(len(Xs), generator=gen)[:64]
        loss = F.cross_entropy(model(Xs[idx]), zs[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (model(X_te).argmax(-1) == z_te).float().mean().item()
    return model, acc

methods = ['scratch', 'probe', 'full', 'lora-1', 'lora-4', 'lora-16']
results = {}
for m in methods:
    runs = [adapt(m, s) for s in range(3)]
    results[m] = dict(acc=[a for _, a in runs], params=trainable(runs[0][0]), model=runs[0][0])
    print(f'{m:8s} accuracy {np.mean(results[m]["acc"]):.3f} ± {np.std(results[m]["acc"]):.3f}'
          f'   paramètres entraînés : {results[m]["params"]:6d}')

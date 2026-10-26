# CELLULE 6 — Tout comparer avec les mêmes sondes, plus deux références.
results, histories = {}, {}
torch.manual_seed(0)
results['aléatoire'] = evaluate(encoder())
results['pixels'] = evaluate(nn.Flatten())             # l'« encodeur » identité
for name in ['invariance', 'vicreg_sans_variance', 'infonce', 'vicreg']:
    t0 = time.time()
    enc, histories[name] = train_joint(name)
    results[name] = evaluate(enc)
    print(f'{name:22s} ({time.time()-t0:.0f} s)')
for name, fn in [('byol', train_byol), ('ijepa', train_ijepa)]:
    t0 = time.time()
    enc, histories[name] = fn()
    results[name] = evaluate(enc)
    print(f'{name:22s} ({time.time()-t0:.0f} s)')
print(f'\n{"méthode":22s} sonde 50 étiq.   sonde 1200  décalé   k-NN   rang eff.')
for k, r in results.items():
    print(f'{k:22s} {r["probe50"]:.3f} ± {r["probe50_sd"]:.3f}   {r["probe_all"]:.3f}       {r["shift_all"]:.3f}    {r["knn"]:.3f}  {r["rank"]:.1f}')

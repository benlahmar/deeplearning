# CELLULE 4 — Pourquoi diviser par √d_k ? Variance des scores et entropie de l'attention.
def entropy(p):
    return float(-(p * np.log(np.clip(p, 1e-300, None))).sum(-1).mean())

r = np.random.default_rng(1)
dims, m = [4, 16, 64, 256, 1024], 32
rows = []
for dk in dims:
    q = r.normal(size=(500, 1, dk)); k = r.normal(size=(500, m, dk))
    s = (q @ np.swapaxes(k, -1, -2))[:, 0]               # 500 requêtes × 32 clés
    rows.append((dk, s.var(), entropy(masked_softmax(s)), entropy(masked_softmax(s / np.sqrt(dk)))))
for dk, v, h_raw, h_scaled in rows:
    print(f'd_k={dk:5d}  Var(q·k)={v:8.1f}  H sans échelle={h_raw:.2f}  H avec 1/√d_k={h_scaled:.2f}')
print('Entropie maximale log(32) =', np.log(m))
plt.semilogx(dims, [x[2] for x in rows], 'o-', label='sans 1/√d_k')
plt.semilogx(dims, [x[3] for x in rows], 's-', label='avec 1/√d_k')
plt.axhline(np.log(m), ls='--', c='gray', label='uniforme')
plt.xlabel('d_k'); plt.ylabel('Entropie moyenne'); plt.legend(); plt.show()

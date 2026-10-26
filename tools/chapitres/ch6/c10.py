# CELLULE 10 — Une valeur aberrante suffit à ruiner la quantification par tenseur.
r = np.random.default_rng(0)
W = r.normal(0, .02, size=(256, 256))
W_out = W.copy(); W_out[3, 7] = 2.0                       # une seule valeur 100 fois plus grande
def rel_err(W, bits, group=None):
    Wq = fake_quant(torch.tensor(W), bits, group).numpy()
    return np.linalg.norm(W - Wq) / np.linalg.norm(W)
for name, M in [('sans valeur aberrante', W), ('avec une valeur aberrante', W_out)]:
    print(f'{name:26s} int8/tenseur {rel_err(M, 8):.4f} | int4/tenseur {rel_err(M, 4):.4f} | int4/groupes 32 {rel_err(M, 4, 32):.4f}')

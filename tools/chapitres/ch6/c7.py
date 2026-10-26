# CELLULE 7 — Efficacité en étiquettes : la valeur du préentraînement dépend de n.
X_big, z_big = sample(1024, seed=5)
label_curve = {m: [] for m in ['scratch', 'probe', 'full', 'lora-4']}
for n in [16, 64, 256, 1024]:
    for m in label_curve:
        label_curve[m].append((n, adapt(m, 0, X_big[:n], z_big[:n])[1]))
    print(n, {m: round(v[-1][1], 3) for m, v in label_curve.items()})
for m, pts in label_curve.items():
    plt.semilogx(*zip(*pts), 'o-', label=m)
plt.xlabel('Exemples étiquetés'); plt.ylabel('Accuracy test'); plt.legend(); plt.show()

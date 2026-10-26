# CELLULE 10 — Courbes d'apprentissage et carte d'attention du modèle appris.
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for position, (_, history, acc) in results.items():
    ax[0].plot(*zip(*history), label=f'{position} ({acc:.0%})')
ax[0].set(xlabel='Mise à jour', ylabel='Entropie croisée', title='Inversion de séquence'); ax[0].legend()
model = results['learned'][0]
with torch.no_grad():
    model(torch.tensor([[3, 1, 4, 1, 5, 9, 2, 6]]))
att = model.blocks[-1].attn.last_attention[0].mean(0)      # moyenne des têtes, dernière couche
ax[1].imshow(att, cmap='Blues'); ax[1].set(xlabel='Clé (position)', ylabel='Requête (position)',
                                          title='Attention, dernière couche')
plt.tight_layout(); plt.show()

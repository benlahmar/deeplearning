# Deep Learning avancé — M2 IA

Support web du cours : huit chapitres, chacun en trois volets (**Cours**, **TD**, **TP**) avec figures interactives, laboratoires dans le navigateur et ateliers Jupyter (corrigé et version à compléter).

| Ch. | Page | Thème |
|---|---|---|
| — | [`index.html`](index.html) | Accueil, programme, méthode |
| 1 | [`Chapitre_1_Reseaux_Profonds.html`](Chapitre_1_Reseaux_Profonds.html) | Réseaux profonds et calcul différentiable |
| 2 | [`Chapitre_2_Optimisation.html`](Chapitre_2_Optimisation.html) | Optimisation et dynamique d’entraînement |
| 3 | [`Chapitre_3_Generalisation.html`](Chapitre_3_Generalisation.html) | Généralisation, régularisation et expérimentation fiable |
| 4 | [`Chapitre_4_Architectures.html`](Chapitre_4_Architectures.html) | Architectures et biais inductifs |
| 5 | [`Chapitre_5_Transformers.html`](Chapitre_5_Transformers.html) | Attention et Transformers |
| 6 | [`Chapitre_6_Transfert.html`](Chapitre_6_Transfert.html) | Préentraînement, transfert et efficacité |
| 7 | [`Chapitre_7_SSL_JEPA.html`](Chapitre_7_SSL_JEPA.html) | Apprentissage auto-supervisé et JEPA |
| 8 | [`Chapitre_8_Energie.html`](Chapitre_8_Energie.html) | Modèles à énergie et ouverture recherche |

## Consulter le cours

- **En ligne** : avec GitHub Pages activé (voir plus bas), le cours est à l'adresse `https://benlahmar.github.io/deeplearning/`.
- **Hors ligne** : ouvrir `index.html` dans un navigateur. Aucune connexion n'est nécessaire, KaTeX est fourni dans `assets/katex/`. Le dossier entier peut être copié sur une clé USB.

Les préférences de lecture, la progression et les notes sont enregistrées dans le navigateur de l'étudiant. Le bouton « Exporter mes notes » en fait une copie texte.

## Ateliers Jupyter

Chaque TP propose deux notebooks : le **corrigé** et la version **à compléter**, avec des `TODO` qui lèvent `NotImplementedError`. Ils sont téléchargeables depuis la page et présents dans [`notebooks/`](notebooks/).

- **Colab** : le bouton « Ouvrir dans Colab » du TP ouvre la version à compléter. Rien à installer. Les étudiants doivent faire *Fichier → Enregistrer une copie dans Drive* pour garder leur travail.
- **En local** :
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cpu   # PyTorch CPU, plus léger
  pip install -r requirements.txt
  jupyter lab notebooks/
  ```

Tout tourne sur CPU. Les durées indiquées dans chaque TP valent pour une machine à 4 cœurs.

## Organisation du dépôt

```
index.html, Chapitre_*.html   pages du cours (autonomes)
assets/carnet.css, carnet.js   charte commune : volets, préférences, progression, notes, figures
assets/katex/                  KaTeX 0.16.9 (licence MIT), pour l'affichage hors ligne des formules
notebooks/                     notebooks exportés depuis les pages (liens Colab)
supports/                      supports PDF d'origine
tools/                         sources et outils de génération (voir ci-dessous)
```

## Modifier le cours

**Chapitres 1 à 4.** La page HTML est elle-même la source : on la modifie directement.

**Chapitres 5 à 8.** Les pages sont générées à partir de `tools/chapitres/chN/` :

| Fichier | Contenu |
|---|---|
| `p1_cours.html`, `p2_td.html`, `p3_tp.html` | corps de la page (en-tête et volets) |
| `c1.py` … `cN.py` | cellules du notebook corrigé, aussi affichées dans le TP |
| `p4_script.js` | figures et laboratoires de la page |
| `config.py` | nom de la page, cellules texte, zones à vider (`TODOS`) pour la version à compléter, données des figures |
| `results_min.json` | résultats d'expériences affichés dans les figures |
| `dump.py` | relance les cellules et régénère les résultats bruts (`chN_results.json`) |

Après une modification :

```bash
python3 tools/build_chapter.py tools/chapitres/ch6   # régénère Chapitre_6_Transfert.html
python3 tools/export_notebooks.py                    # met à jour notebooks/ (liens Colab)
```

Ne modifiez pas directement `Chapitre_5` à `Chapitre_8` : la génération suivante écraserait les changements.

Le JSON des figures (`results_min.json`) est une version arrondie et allégée des résultats bruts produits par `dump.py`. Si vous changez une expérience, relancez `dump.py` depuis le dossier du chapitre, puis reportez les nouvelles valeurs.

## Vérifier avant de publier

```bash
python3 tools/run_notebook.py Chapitre_6_Transfert.html complete   # le corrigé doit aller au bout
node tools/check_site.js                                           # nécessite Playwright (npm i playwright)
```

`check_site.js` ouvre chaque page avec le réseau coupé. Il signale :
- les erreurs JavaScript ;
- les formules KaTeX en erreur ou qui débordent ;
- les liens locaux cassés ;
- un téléchargement de notebook qui échoue ;
- toute requête externe.

## Publier avec GitHub Pages

*Settings → Pages → Build and deployment* : choisir **Deploy from a branch**, puis la branche `main` et le dossier `/ (root)`. Le site est en ligne au bout d'une à deux minutes.

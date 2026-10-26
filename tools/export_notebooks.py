"""Exporte les notebooks intégrés aux pages (corrigé et à compléter) dans notebooks/.

    python3 tools/export_notebooks.py

Les pages restent la source : relancer ce script après toute modification d'un atelier,
pour que les liens « Ouvrir dans Colab » pointent vers la version à jour.
"""
import json, re, glob, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.makedirs(os.path.join(ROOT, 'notebooks'), exist_ok=True)
for page in sorted(glob.glob(os.path.join(ROOT, 'Chapitre_*.html'))):
    s = open(page, encoding='utf-8').read()
    nbs = json.loads(re.search(r'<script type="application/json" id="notebooks">(.*?)</script>', s, re.S).group(1))
    for kind, name in re.findall(r'data-notebook="(\w+)" data-file="([^"]+)"', s):
        with open(os.path.join(ROOT, 'notebooks', name), 'w', encoding='utf-8') as f:
            json.dump(nbs[kind], f, ensure_ascii=False, indent=1); f.write('\n')
        print('ok', 'notebooks/' + name)

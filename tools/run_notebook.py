"""Exécute de bout en bout le notebook intégré à une page, sans Jupyter.

    python3 tools/run_notebook.py Chapitre_6_Transfert.html complete   # corrigé : doit aller au bout
    python3 tools/run_notebook.py Chapitre_6_Transfert.html starter    # à compléter : s'arrête au premier TODO
"""
import json, re, sys, time, os
os.environ['MPLBACKEND'] = 'Agg'
s = open(sys.argv[1], encoding='utf-8').read()
nb = json.loads(re.search(r'<script type="application/json" id="notebooks">(.*?)</script>', s, re.S).group(1))[sys.argv[2] if len(sys.argv) > 2 else 'complete']
g, ok = {}, True
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    print(f'--- cellule {i}: {src.splitlines()[0][:70]}')
    t = time.time()
    try:
        exec(src, g)
    except NotImplementedError as e:
        print('NotImplementedError :', e); ok = False; break
    except Exception as e:
        print('ERREUR', type(e).__name__, e); ok = False; break
    print(f'    ({time.time() - t:.1f} s)')
sys.exit(0 if ok else 1)

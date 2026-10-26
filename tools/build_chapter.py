"""Assemble une page de chapitre à partir de ses sources.

    python3 tools/build_chapter.py tools/chapitres/ch6            # écrit Chapitre_6_Transfert.html à la racine
    python3 tools/build_chapter.py tools/chapitres/ch6 sortie.html

Sources d'un chapitre : p1_cours.html + p2_td.html + p3_tp.html (le corps de la page),
c1.py … cN.py (cellules du notebook corrigé, insérées dans le TP à la place de @@Ci@@),
p4_script.js (figures et laboratoires ; @@CLE@@ remplacé par config.DATA) et config.py :
PAGE, OUT, NCELLS, TODOS (zones à vider dans le notebook à compléter), MD (cellules texte),
DATA, REPLACE (substitutions finales), RAISE_SHORT (optionnel).
"""
import re, json, html, sys, os, importlib.util
S = os.path.abspath(sys.argv[1])
spec = importlib.util.spec_from_file_location('cfg', f'{S}/config.py'); cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', cfg.OUT)
R = lambda f: open(f'{S}/{f}', encoding='utf-8').read()
n = cfg.NCELLS
cells = {i: R(f'c{i}.py').rstrip() + '\n' for i in range(1, n + 1)}

def todo(src, start_pat, end_pat, msg, indent='    '):
    m = re.search(start_pat + r'(.*?)' + end_pat, src, re.S)
    assert m, start_pat
    short = msg.split(' :')[0] if getattr(cfg, 'RAISE_SHORT', False) else msg
    body = f'{indent}# TODO : {msg}\n{indent}raise NotImplementedError("{short}")\n'
    return src[:m.start(1)] + body + src[m.end(1):]
starter = dict(cells)
for i, sp, ep, msg, *ind in cfg.TODOS:
    starter[i] = todo(starter[i], sp, ep, msg, *(ind or []))
    assert 'NotImplementedError' in starter[i]; compile(starter[i], f's{i}', 'exec')

def lines(t):
    p = t.split('\n'); return [l + '\n' for l in p[:-1]] + ([p[-1]] if p[-1] else [])
def notebook(cs, kind):
    nb = []
    for i in range(1, n + 1):
        nb.append({"cell_type": "markdown", "metadata": {}, "id": f"{cfg.PAGE}-m{i}", "source": lines(cfg.MD[i].format(kind=kind))})
        nb.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "id": f"{cfg.PAGE}-c{i}", "source": lines(cs[i])})
    return {"cells": nb, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
nbs = {'complete': notebook(cells, 'corrigé'), 'starter': notebook(starter, 'à compléter')}

page = R('p1_cours.html') + R('p2_td.html') + R('p3_tp.html')
for a, b in getattr(cfg, 'REPLACE', []):
    assert page.count(a) == 1, a; page = page.replace(a, b)
for i in range(1, n + 1):
    page = page.replace(f'@@C{i}@@', html.escape(cells[i].rstrip()))
assert '@@C' not in page
js = R('p4_script.js')
for k, v in cfg.DATA.items():
    js = js.replace(k, v)
assert '@@' not in js, re.findall('@@\\w+@@', js)
payload = json.dumps(nbs, ensure_ascii=False).replace('</', '<\\/')
page += f'\n<script type="application/json" id="notebooks">{payload}</script>\n<script src="assets/carnet.js"></script>\n<script>\n{js}\n</script>\n</body></html>\n'
open(out, 'w', encoding='utf-8').write(page)
print('ok', os.path.relpath(out), len(page))

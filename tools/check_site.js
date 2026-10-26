// Vérifie toutes les pages dans Chromium (Playwright), réseau coupé :
// erreurs JS, formules KaTeX en erreur ou qui débordent, liens locaux cassés, téléchargement des notebooks.
//   node tools/check_site.js
const {chromium}=require('playwright');const fs=require('fs');const path=require('path');
(async()=>{const D=path.join(__dirname,'..')+'/';const b=await chromium.launch();let bad=0;
const pages=fs.readdirSync(D).filter(f=>f.endsWith('.html')).sort();
for(const f of pages){const c=await b.newContext({viewport:{width:1400,height:900},acceptDownloads:true});const ext=[];
 await c.route(/^https?:/,r=>{ext.push(r.request().url());r.abort()});
 const p=await c.newPage();const errs=[];p.on('pageerror',e=>errs.push(e.message));p.on('console',m=>{if(m.type()==='error'&&!/ERR_FAILED/.test(m.text()))errs.push(m.text())});
 await p.goto('file://'+D+f);await p.waitForTimeout(1200);
 const info=await p.evaluate(async()=>{const r={};r.katex=document.querySelectorAll('.katex').length;r.kerr=document.querySelectorAll('.katex-error').length;
  const links=[...document.querySelectorAll('a[href]')].map(a=>a.getAttribute('href')).filter(h=>!/^(https?:|#|mailto:)/.test(h));r.links=[...new Set(links)];
  r.over=[];for(const rl of document.querySelectorAll('.rail')){rl.click();await new Promise(z=>setTimeout(z,150));document.querySelectorAll('.chapter.active details').forEach(d=>d.open=true);document.querySelectorAll('.chapter.active .phase').forEach(ph=>ph.hidden=false);document.querySelectorAll('.chapter.active .formula').forEach(fm=>{if(fm.scrollWidth>fm.clientWidth+1)r.over.push((fm.closest('section')||{}).id+': '+fm.textContent.slice(0,30))})}
  r.colab=document.querySelectorAll('a.colab').length;return r});
 let dl='';const btn=await p.$('[data-notebook="starter"]');if(btn){await p.evaluate(()=>{document.querySelectorAll('.rail')[2].click()});await p.waitForTimeout(200);const [d]=await Promise.all([p.waitForEvent('download'),btn.click()]);dl=d.suggestedFilename()+' ('+JSON.parse(fs.readFileSync(await d.path(),'utf8')).cells.length+' cellules)'}
 const missing=info.links.map(h=>decodeURI(h.split('#')[0])).filter(h=>h&&!fs.existsSync(D+h));
 const ko=info.kerr||info.over.length||missing.length||errs.length;bad+=!!ko;
 console.log((ko?'✗ ':'✓ ')+f.padEnd(34),'formules',info.katex,'erreurs KaTeX',info.kerr,'débordements',info.over.length?info.over:0,'Colab',info.colab,dl,'liens cassés',missing.length?missing:0,'erreurs JS',errs.length?errs:0,'requêtes externes',ext.length);await c.close()}
await b.close();process.exit(bad?1:0)})();

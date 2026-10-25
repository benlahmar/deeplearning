// Le carnet — comportements communs à tous les chapitres.
// La page déclare <body data-page="ch5">. Les réglages de lecture sont partagés entre les pages.
(function(){
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const page=document.body.dataset.page||'page', key='carnet-'+page+'-v1', prefsKey='carnet-prefs-v1';
let state={}, prefs=null;
try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
try{prefs=JSON.parse(localStorage.getItem(prefsKey)||'null')}catch(e){}
function save(){try{localStorage.setItem(key,JSON.stringify(state));return true}catch(e){return false}}
const fmt=(x,d=2)=>Number(x).toLocaleString('fr-FR',{minimumFractionDigits:d,maximumFractionDigits:d});
const smooth=()=>document.body.classList.contains('still')?'auto':'smooth';

// ---------- Réglages de lecture (communs à toutes les pages) ----------
function appearance(){
 const b=document.body;b.classList.remove('dark','contrast');if($('#theme').value)b.classList.add($('#theme').value);
 document.documentElement.style.setProperty('--size',$('#size').value+'px');
 ['airy','sans'].forEach(id=>b.classList.toggle(id,$('#'+id).checked));
 b.classList.toggle('motion',$('#animate').checked);b.classList.toggle('still',!$('#animate').checked);
 try{localStorage.setItem(prefsKey,JSON.stringify({theme:$('#theme').value,size:$('#size').value,airy:$('#airy').checked,sans:$('#sans').checked,animate:$('#animate').checked}))}catch(e){}
 document.dispatchEvent(new CustomEvent('carnet:appearance'));
}
if($('#settings')){
 $('#prefs').onclick=()=>{const open=$('#settings').hidden;$('#settings').hidden=!open;$('#prefs').setAttribute('aria-expanded',open)};
 document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('#settings').hidden){$('#settings').hidden=true;$('#prefs').setAttribute('aria-expanded','false');$('#prefs').focus()}});
 if(prefs){$('#theme').value=prefs.theme||'';$('#size').value=prefs.size||'18';['airy','sans'].forEach(id=>$('#'+id).checked=!!prefs[id]);$('#animate').checked=prefs.animate!==false}
 else if(matchMedia('(prefers-color-scheme: dark)').matches)$('#theme').value='dark';
 if(matchMedia('(prefers-reduced-motion: reduce)').matches&&!prefs)$('#animate').checked=false;
 $$('#settings input, #settings select').forEach(el=>el.onchange=appearance);appearance();
}
if($('#focus'))$('#focus').onclick=()=>{const on=document.body.classList.toggle('focus');$('#focus').setAttribute('aria-pressed',on)};
$$('.banner-toggle').forEach(btn=>btn.onclick=()=>{const paused=btn.closest('.section-banner').classList.toggle('paused');btn.setAttribute('aria-pressed',String(paused));btn.textContent=paused?'Reprendre l’animation':'Mettre l’animation en pause'});

// ---------- Impression ----------
let beforePrint=[];
window.addEventListener('beforeprint',()=>{beforePrint=$$('details').map(d=>d.open);$$('details').forEach(d=>d.open=true)});
window.addEventListener('afterprint',()=>$$('details').forEach((d,i)=>d.open=beforePrint[i]));
if($('#print'))$('#print').onclick=()=>window.print();

// ---------- Progression ----------
const marks=$$('[data-done]');
function progress(){if(!$('#progressText'))return;const n=marks.filter(el=>el.checked).length;$('#progressText').textContent=`${n} / ${marks.length} repères validés`;$('#bar').style.width=(marks.length?n/marks.length*100:0)+'%'}
marks.forEach(el=>{el.checked=!!state['done:'+el.dataset.done];el.onchange=()=>{state['done:'+el.dataset.done]=el.checked;save();progress()}});progress();

// ---------- Notes, étapes, téléchargements ----------
$$('textarea[data-note]').forEach(el=>{el.value=state['note:'+el.id]||'';el.oninput=()=>{state['note:'+el.id]=el.value;const s=$('#notesStatus');if(s)s.textContent=save()?'Notes enregistrées dans ce navigateur.':'Stockage indisponible : exportez vos notes avant de quitter.'}});
const showPhase={};
$$('[data-phases]').forEach(group=>{
 const name=group.dataset.phases, sections=$$(`.phase[data-group="${name}"]`), buttons=$$(`[data-phase-of="${name}"]`);
 const show=i=>{showPhase[name]=show;sections.forEach((s,n)=>s.hidden=n!==i);buttons.forEach(b=>{if(+b.dataset.phase===i&&b.closest('[data-phases]'))b.setAttribute('aria-current','step');else b.removeAttribute('aria-current')});state['phase:'+name]=i;save()};
 buttons.forEach(b=>b.onclick=()=>{show(+b.dataset.phase);if(!b.closest('[data-phases]')){const t=sections[+b.dataset.phase];t.setAttribute('tabindex','-1');t.focus()}});
 show(Math.min(sections.length-1,Math.max(0,Number(state['phase:'+name])||0)));
});
function download(name,type,text){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
$$('[data-notebook]').forEach(b=>b.onclick=()=>{const nb=JSON.parse($('#notebooks').textContent)[b.dataset.notebook];download(b.dataset.file,'application/x-ipynb+json',JSON.stringify(nb,null,1))});
if($('#exportNotes'))$('#exportNotes').onclick=()=>download(`Notes_${page}.txt`,'text/plain;charset=utf-8',document.title+' — Notes personnelles\n\n'+$$('textarea[data-note]').map(el=>{const l=document.querySelector(`label[for="${el.id}"]`);return (l?l.textContent:el.id)+'\n'+(el.value||'(non renseigné)')}).join('\n\n')+'\n\n'+($('#progressText')?$('#progressText').textContent:''));

// ---------- Volets : un seul ouvert, adresse synchronisée, recherche dans les volets fermés ----------
const panels=$$('.chapter'), labels=panels.map(p=>p.querySelector('.rail-word').textContent);
const untilFound='onbeforematch' in document.body;
function hide(content){if(untilFound)content.setAttribute('hidden','until-found');else content.hidden=true}
function activatePanel(index,hash){panels.forEach((panel,i)=>{const active=i===index,rail=panel.querySelector('.rail'),content=panel.querySelector('.content');panel.classList.toggle('active',active);rail.setAttribute('aria-expanded',String(active));rail.setAttribute('aria-label',(active?'Volet ouvert : ':'Ouvrir ')+labels[i]);if(active)content.removeAttribute('hidden');else hide(content)});if(hash!==false)history.replaceState(null,'','#'+(hash||panels[index].id));document.dispatchEvent(new CustomEvent('carnet:panel',{detail:index}))}
function reveal(id,focus){const target=id&&document.getElementById(id);if(!target)return false;const panel=target.closest('.chapter');if(!panel)return false;const ph=target.closest('.phase');if(ph&&ph.hidden&&showPhase[ph.dataset.group])showPhase[ph.dataset.group]($$(`.phase[data-group="${ph.dataset.group}"]`).indexOf(ph));activatePanel(panels.indexOf(panel),target===panel?panel.id:id);if(target!==panel){target.scrollIntoView({behavior:smooth(),block:'start'});if(focus){target.setAttribute('tabindex','-1');target.focus({preventScroll:true})}}return true}
panels.forEach((panel,i)=>{
 const rail=panel.querySelector('.rail'),content=panel.querySelector('.content');
 rail.onclick=()=>{activatePanel(i);$('#readingStatus').textContent=labels[i]+' · reprendre la lecture'};
 rail.onkeydown=e=>{let next=i;if(e.key==='ArrowRight'||e.key==='ArrowDown')next=(i+1)%panels.length;else if(e.key==='ArrowLeft'||e.key==='ArrowUp')next=(i+panels.length-1)%panels.length;else if(e.key==='Home')next=0;else if(e.key==='End')next=panels.length-1;else return;e.preventDefault();activatePanel(next);panels[next].querySelector('.rail').focus()};
 content.addEventListener('beforematch',()=>activatePanel(i));
 const top=document.createElement('button');top.className='jump-top';top.textContent='↑ Début du volet';top.onclick=()=>content.scrollTo({top:0,behavior:smooth()});content.append(top);
 content.addEventListener('scroll',()=>{const range=content.scrollHeight-content.clientHeight;$('#readingStatus').textContent=labels[i]+' · lecture '+(range>0?Math.round(content.scrollTop/range*100):0)+' %'});
});
$$('a[href^="#"]').forEach(a=>{if(a.classList.contains('skip'))return;a.addEventListener('click',e=>{const id=a.getAttribute('href').slice(1);if(document.getElementById(id)&&document.getElementById(id).closest('.chapter')){e.preventDefault();reveal(id,true)}})});
window.addEventListener('hashchange',()=>reveal(decodeURIComponent(location.hash.slice(1)),true));
if(panels.length&&!reveal(decodeURIComponent(location.hash.slice(1))))activatePanel(0,false);

// ---------- Formules ----------
function renderMath(){if(window.renderMathInElement)renderMathInElement(document.body,{delimiters:[{left:'\\[',right:'\\]',display:true},{left:'\\(',right:'\\)',display:false}],ignoredTags:['script','noscript','style','textarea','pre','code','option'],throwOnError:false})}
document.addEventListener('DOMContentLoaded',()=>{renderMath();const id=decodeURIComponent(location.hash.slice(1)),t=id&&document.getElementById(id);if(t&&!t.classList.contains('chapter'))t.scrollIntoView({block:'start'})});

// ---------- Outils pour les figures des pages ----------
const NS='http://www.w3.org/2000/svg';
function el(tag,attrs={},parent){const n=document.createElementNS(NS,tag);for(const k in attrs)n.setAttribute(k,attrs[k]);if(parent)parent.append(n);return n}
function text(parent,x,y,str,cls='tick',extra={}){const e=Object.assign({},extra),st=[];if(e.fill){st.push(`fill:${e.fill}`);delete e.fill}if(e['text-anchor']){st.push(`text-anchor:${e['text-anchor']}`);delete e['text-anchor']}if(st.length)e.style=st.join(';');const t=el('text',Object.assign({x,y,class:cls},e),parent);t.textContent=str;return t}
function rng(seed){let s=seed>>>0||1;return()=>{s=(1664525*s+1013904223)>>>0;return(s+.5)/4294967296}}
function gauss(r){return()=>{const u=r(),v=r();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}}
function softmax(v){const m=Math.max(...v.filter(Number.isFinite));const e=v.map(x=>Number.isFinite(x)?Math.exp(x-m):0);const s=e.reduce((a,b)=>a+b,0);return e.map(x=>s>0?x/s:NaN)}
// Graphique en lignes : opts = {w,h,x:[min,max],y:[min,max],xlog,ylog,xticks,yticks,xlabel,ylabel,series:[{pts,cls,label}],fmtx,fmty}
function plot(svg,o){
 svg.innerHTML='';const W=o.w||620,H=o.h||300,m={l:58,r:18,t:16,b:44};svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
 const tx=v=>{const a=o.xlog?Math.log10(v):v,lo=o.xlog?Math.log10(o.x[0]):o.x[0],hi=o.xlog?Math.log10(o.x[1]):o.x[1];return m.l+(a-lo)/(hi-lo)*(W-m.l-m.r)};
 const ty=v=>{const a=o.ylog?Math.log10(v):v,lo=o.ylog?Math.log10(o.y[0]):o.y[0],hi=o.ylog?Math.log10(o.y[1]):o.y[1];return H-m.b-(a-lo)/(hi-lo)*(H-m.t-m.b)};
 (o.yticks||[]).forEach(v=>{el('line',{x1:m.l,x2:W-m.r,y1:ty(v),y2:ty(v),class:'gridline'},svg);text(svg,m.l-8,ty(v)+4,(o.fmty||String)(v),'tick',{'text-anchor':'end'})});
 (o.xticks||[]).forEach(v=>{el('line',{x1:tx(v),x2:tx(v),y1:H-m.b,y2:H-m.b+5,class:'ax'},svg);text(svg,tx(v),H-m.b+18,(o.fmtx||String)(v),'tick',{'text-anchor':'middle'})});
 el('path',{d:`M${m.l} ${m.t}V${H-m.b}H${W-m.r}`,class:'ax'},svg);
 if(o.xlabel)text(svg,(m.l+W-m.r)/2,H-6,o.xlabel,'axlabel',{'text-anchor':'middle'});
 if(o.ylabel)text(svg,14,(m.t+H-m.b)/2,o.ylabel,'axlabel',{'text-anchor':'middle',transform:`rotate(-90 14 ${(m.t+H-m.b)/2})`});
 (o.series||[]).forEach(s=>{const pts=s.pts.filter(p=>Number.isFinite(p[1])&&(!o.ylog||p[1]>0));if(!pts.length)return;el('path',{d:pts.map((p,i)=>(i?'L':'M')+tx(p[0]).toFixed(1)+' '+ty(Math.min(Math.max(p[1],o.y[0]),o.y[1])).toFixed(1)).join(''),class:s.cls},svg);if(s.dots)pts.forEach(p=>el('circle',{cx:tx(p[0]),cy:ty(p[1]),r:3,class:s.dots},svg))});
 return {tx,ty,m,W,H};
}
window.Carnet={$,$$,fmt,el,text,rng,gauss,softmax,plot,state,save,reveal};
})();

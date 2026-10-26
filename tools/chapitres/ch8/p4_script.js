// Chapitre 8 — figures et laboratoires.
(function(){
const {$,fmt,el,text,rng,gauss,softmax,plot}=window.Carnet;
const R=@@RESULTS@@;
const MODES=Array.from({length:8},(_,k)=>[2*Math.cos(2*Math.PI*k/8),2*Math.sin(2*Math.PI*k/8)]);
// Mélange de gaussiennes : log-densité et score exacts, variance totale v.
function gmm(x,y,v){let lw=MODES.map(([a,b])=>-((x-a)**2+(y-b)**2)/(2*v));const m=Math.max(...lw),w=lw.map(l=>Math.exp(l-m)),s=w.reduce((p,q)=>p+q);
 const logp=m+Math.log(s/8)-Math.log(2*Math.PI*v);let gx=0,gy=0;MODES.forEach(([a,b],k)=>{gx+=w[k]/s*(a-x)/v;gy+=w[k]/s*(b-y)/v});return {logp,gx,gy}}
const panel=(svg,x0,y0,size,lim)=>({sx:v=>x0+(v+lim)/(2*lim)*size,sy:v=>y0+size-(v+lim)/(2*lim)*size});

// ---------- 8.1 · Deux réponses plausibles ----------
const BI=(()=>{const r=rng(3),g=gauss(r);return Array.from({length:220},()=>{const x=2*r()-1,z=r()<.5?-1:1;return [x,z*x*x+.06*g()]})})();
function drawBimodal(){const x=+$('#bX').value,beta=+$('#bBeta').value;$('#bXOut').textContent=fmt(x);$('#bBetaOut').textContent=fmt(beta,1);
 const svg=$('#bimodalPlot');svg.innerHTML='';const P=panel(svg,40,20,230,1.25);
 el('rect',{x:40,y:20,width:230,height:230,class:'gridline',fill:'none'},svg);
 BI.forEach(([a,b])=>el('circle',{cx:P.sx(a),cy:P.sy(b),r:2.6,class:'f-a','fill-opacity':.55},svg));
 el('line',{x1:P.sx(-1.25),x2:P.sx(1.25),y1:P.sy(0),y2:P.sy(0),class:'s-b'},svg);text(svg,P.sx(.15),P.sy(0)-6,'régression : E[y | x] ≈ 0','tick',{fill:'var(--ochre)'});
 el('line',{x1:P.sx(x),x2:P.sx(x),y1:20,y2:250,class:'s-d'},svg);text(svg,155,272,'x','axlabel',{'text-anchor':'middle'});text(svg,22,135,'y','axlabel');
 const ys=[];for(let y=-1.25;y<=1.25;y+=.01)ys.push(y);const E=ys.map(y=>Math.min((y-x*x)**2,(y+x*x)**2)),p=softmax(E.map(e=>-beta*e*20));const pm=Math.max(...p);
 const x0=330,w=280,sy=y=>20+230-(y+1.25)/2.5*230;
 el('path',{d:`M${x0} 20V250H${x0+w}`,class:'ax'},svg);
 el('path',{d:ys.map((y,i)=>(i?'L':'M')+(x0+Math.min(1,E[i]/1.6)*w).toFixed(1)+' '+sy(y).toFixed(1)).join(''),class:'s-a'},svg);
 el('path',{d:ys.map((y,i)=>(i?'L':'M')+(x0+p[i]/pm*w*.9).toFixed(1)+' '+sy(y).toFixed(1)).join('')+`L${x0} ${sy(1.25)}L${x0} ${sy(-1.25)}Z`,fill:'var(--ochre)','fill-opacity':.25,stroke:'var(--ochre)'},svg);
 [-1,0,1].forEach(v=>text(svg,x0-6,sy(v)+4,fmt(v,0),'tick',{'text-anchor':'end'}));
 text(svg,x0+w/2,272,'bleu : énergie E(x, y) · ocre : p(y | x) ∝ e^(−βE)','axlabel',{'text-anchor':'middle'});
 $('#bReadout').textContent=`Minima de l’énergie en y = ±${fmt(x*x)} ; la régression prédit ≈ 0, une valeur jamais observée pour x = ${fmt(x)}.`}
['bX','bBeta'].forEach(id=>$('#'+id).oninput=drawBimodal);drawBimodal();

// ---------- 8.3 · Façonner l'énergie ----------
(()=>{const svg=$('#shapeFig'),data=[-1.6,-1.2,.9,1.3],neg=[-.2,2.4],xs=[];for(let x=-3;x<=3;x+=.03)xs.push(x);
 const before=x=>.6+.05*x*x,after=x=>{let s=0;data.forEach(d=>s+=Math.exp(-((x-d)**2)/.12));return 1.2-1.0*Math.min(1,s)+.04*x*x};
 const sx=x=>40+(x+3)/6*560,sy=v=>200-v*120;
 el('path',{d:xs.map((x,i)=>(i?'L':'M')+sx(x).toFixed(1)+' '+sy(before(x)).toFixed(1)).join(''),class:'s-d'},svg);
 el('path',{d:xs.map((x,i)=>(i?'L':'M')+sx(x).toFixed(1)+' '+sy(after(x)).toFixed(1)).join(''),class:'s-a'},svg);
 el('path',{d:`M40 205H600`,class:'ax'},svg);
 const defs=el('defs',{},svg);[['sarrA','var(--blue)'],['sarrB','var(--ochre)']].forEach(([id,c])=>{const mk=el('marker',{id,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto'},defs);el('path',{d:'M0 0L10 5L0 10z',style:'fill:'+c},mk)});
 data.forEach(d=>{el('circle',{cx:sx(d),cy:205,r:6,class:'f-a'},svg);el('line',{x1:sx(d),x2:sx(d),y1:sy(before(d))-28,y2:sy(before(d))-4,stroke:'var(--blue)','stroke-width':2,'marker-end':'url(#sarrA)'},svg)});
 neg.forEach(d=>{el('circle',{cx:sx(d),cy:205,r:6,class:'f-b'},svg);el('line',{x1:sx(d),x2:sx(d),y1:sy(after(d))+30,y2:sy(after(d))+6,stroke:'var(--ochre)','stroke-width':2,'marker-end':'url(#sarrB)'},svg)});
 text(svg,sx(-3)+4,sy(before(-3))-8,'avant','tick');text(svg,sx(2.6),sy(after(2.6))-8,'après','tick',{fill:'var(--blue)'});
 text(svg,320,225,'bleu : données (énergie abaissée) · ocre : échantillons du modèle (énergie relevée)','axlabel',{'text-anchor':'middle'});})();

// ---------- 8.4 · Langevin ----------
const LV=.25*.25;let parts=[],lTimer=null,lRng=rng(9),lG=gauss(lRng);
function lReset(){parts=Array.from({length:300},()=>[6*lRng()-3,6*lRng()-3]);lSteps=0;drawLangevin()}
let lSteps=0;
function drawLangevin(){const svg=$('#langevinPlot');svg.innerHTML='';const P=panel(svg,20,10,310,3.2),n=46,c=310/n;
 let mx=-Infinity;const dens=[];for(let i=0;i<n;i++)for(let j=0;j<n;j++){const x=-3.2+(j+.5)*6.4/n,y=3.2-(i+.5)*6.4/n,d=Math.exp(gmm(x,y,LV).logp);dens.push(d);mx=Math.max(mx,d)}
 dens.forEach((d,k)=>el('rect',{x:20+(k%n)*c,y:10+Math.floor(k/n)*c,width:c+.4,height:c+.4,fill:'var(--blue)','fill-opacity':(.04+.5*Math.sqrt(d/mx)).toFixed(3)},svg));
 parts.forEach(([x,y])=>el('circle',{cx:P.sx(x),cy:P.sy(y),r:2.4,class:'f-b'},svg));
 const counts=new Array(8).fill(0);let near=0,sp=0;parts.forEach(([x,y])=>{let best=0,bd=1e9;MODES.forEach(([a,b],k)=>{const d=Math.hypot(x-a,y-b);if(d<bd){bd=d;best=k}});if(bd<.75){near++;counts[best]++;sp+=bd*bd}});
 const x0=360,w=260;text(svg,x0,26,'Répartition des particules entre les modes','axlabel');
 counts.forEach((c2,k)=>{const h=Math.min(1,c2/(parts.length/8)/2)*150;el('rect',{x:x0+k*32,y:190-h,width:24,height:h,fill:'var(--ochre)','fill-opacity':.8},svg);text(svg,x0+k*32+12,206,String(k+1),'tick',{'text-anchor':'middle'})});
 el('line',{x1:x0,x2:x0+w,y1:190-75,y2:190-75,class:'s-d'},svg);text(svg,x0+w,190-80,'répartition idéale','tick',{'text-anchor':'end'});
 text(svg,x0,240,`${lSteps} pas · ${fmt(100*near/parts.length,0)} % près d’un mode`,'tick');text(svg,x0,262,`dispersion autour du mode : ${near?fmt(Math.sqrt(sp/near/2),3):'—'} (cible : 0,250)`,'tick');
 $('#lReadout').textContent=$('#lNoise').checked?'Échantillonnage : les particules doivent se répartir entre les huit modes avec la bonne dispersion.':'Sans bruit : descente de gradient, les particules s’écrasent au fond des puits (dispersion → 0).'}
function lStep(){const eps=Math.pow(10,+$('#lEps').value),noise=$('#lNoise').checked;for(let s=0;s<5;s++){parts=parts.map(([x,y])=>{const {gx,gy}=gmm(x,y,LV);let nx=x+.5*eps*gx,ny=y+.5*eps*gy;if(noise){nx+=Math.sqrt(eps)*lG();ny+=Math.sqrt(eps)*lG()}return [Math.max(-4,Math.min(4,nx)),Math.max(-4,Math.min(4,ny))]});lSteps++}drawLangevin()}
function lStop(){if(lTimer){clearInterval(lTimer);lTimer=null}$('#lRun').textContent='Lancer'}
$('#lRun').onclick=()=>{if(lTimer){lStop();return}if(!document.body.classList.contains('motion')){for(let i=0;i<60;i++)lStep();return}$('#lRun').textContent='Pause';lTimer=setInterval(()=>{lStep();if(lSteps>2000)lStop()},60)};
$('#lReset').onclick=()=>{lStop();lReset()};$('#lEps').oninput=()=>{$('#lEpsOut').textContent=fmt(Math.pow(10,+$('#lEps').value),4)};$('#lEps').oninput();$('#lNoise').onchange=drawLangevin;lReset();
document.addEventListener('carnet:panel',lStop);

// ---------- 8.6 · Champ de score ----------
function drawScore(){const s=Math.pow(10,+$('#sSig').value);$('#sSigOut').textContent=fmt(s,3);const v=.15*.15+s*s;
 const svg=$('#scorePlot');svg.innerHTML='';const P=panel(svg,20,10,310,3.2),n=40,c=310/n;let mx=0;const dens=[];
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){const x=-3.2+(j+.5)*6.4/n,y=3.2-(i+.5)*6.4/n,d=Math.exp(gmm(x,y,v).logp);dens.push(d);mx=Math.max(mx,d)}
 dens.forEach((d,k)=>el('rect',{x:20+(k%n)*c,y:10+Math.floor(k/n)*c,width:c+.4,height:c+.4,fill:'var(--blue)','fill-opacity':(.03+.5*Math.sqrt(d/mx)).toFixed(3)},svg));
 const defs=el('defs',{},svg),mk=el('marker',{id:'qa',viewBox:'0 0 10 10',refX:8,refY:5,markerWidth:5,markerHeight:5,orient:'auto'},defs);el('path',{d:'M0 0L10 5L0 10z',class:'f-b'},mk);
 for(let i=0;i<15;i++)for(let j=0;j<15;j++){const x=-2.8+j*.4,y=-2.8+i*.4,{gx,gy}=gmm(x,y,v),nrm=Math.hypot(gx,gy),dens2=Math.exp(gmm(x,y,v).logp)/mx;if(nrm<1e-9)continue;const L=Math.min(14,4+4*Math.log1p(nrm));
  el('line',{x1:P.sx(x),y1:P.sy(y),x2:P.sx(x)+L*gx/nrm,y2:P.sy(y)-L*gy/nrm,stroke:'var(--ochre)','stroke-width':1.3,'stroke-opacity':(.25+.75*Math.min(1,Math.sqrt(dens2)*3)).toFixed(2),'marker-end':'url(#qa)'},svg)}
 const x0=360;text(svg,x0,30,'Lecture','axlabel');[`σ = ${fmt(s,3)} ; densité bruitée p_σ = p ∗ N(0, σ²)`,'fond bleu : densité ; flèches : ∇ log p_σ','opacité des flèches : densité locale','loin des modes et pour σ petit :','le score n’est jamais observé'].forEach((t2,k)=>text(svg,x0,56+k*20,t2,'tick'));}
$('#sSig').oninput=drawScore;drawScore();

// ---------- Nuages de points (résultats réels) ----------
function scatter(svg,x0,y0,size,pts,title,cls){const P=panel(svg,x0,y0,size,3);el('rect',{x:x0,y:y0,width:size,height:size,class:'gridline',fill:'none'},svg);MODES.forEach(([a,b])=>el('circle',{cx:P.sx(a),cy:P.sy(b),r:size*.15/6*3,fill:'none',stroke:'var(--line)'},svg));pts.forEach(([a,b])=>el('circle',{cx:P.sx(a),cy:P.sy(b),r:1.6,class:cls||'f-a','fill-opacity':.7},svg));text(svg,x0+size/2,y0+size+16,title,'tick',{'text-anchor':'middle'})}
(()=>{const svg=$('#diffPlot');svg.setAttribute('viewBox','0 0 640 150');[199,150,100,50,0].forEach((t,i)=>scatter(svg,8+i*127,6,116,R.snaps[String(t)],'t = '+t,'f-b'))})();
(()=>{const svg=$('#ebmEnergy');svg.setAttribute('viewBox','0 0 300 300');const n=61,c=270/n,mx=Math.max(...R.energy);
 R.energy.forEach((e,k)=>{const i=Math.floor(k/n),j=k%n;el('rect',{x:15+j*c,y:15+(n-1-i)*c,width:c+.3,height:c+.3,fill:'var(--blue)','fill-opacity':(.75*(1-Math.min(1,e/(mx*.6)))).toFixed(3)},svg)});
 const P=panel(svg,15,15,270,3);R.samples['EBM, Langevin'].forEach(([a,b])=>el('circle',{cx:P.sx(a),cy:P.sy(b),r:1.5,class:'f-b','fill-opacity':.8},svg));
 plot($('#ebmHist'),{w:300,h:300,x:[0,2900],y:[Math.min(...R.hist.map(h=>Math.min(h[1],h[2])))-.2,Math.max(...R.hist.map(h=>Math.max(h[1],h[2])))+.2],xticks:[0,1000,2000],yticks:[],xlabel:'mise à jour',ylabel:'énergie moyenne',series:[{pts:R.hist.map(h=>[h[0],h[1]]),cls:'s-a'},{pts:R.hist.map(h=>[h[0],h[2]]),cls:'s-b'}]})})();
(()=>{const svg=$('#comparePlot');svg.setAttribute('viewBox','0 0 640 450');const keys=Object.keys(R.samples);keys.forEach((k,i)=>scatter(svg,14+(i%3)*212,8+Math.floor(i/3)*222,190,R.samples[k],k,k.includes('sans bruit')||k.includes('un seul')?'f-b':'f-a'));
 $('#compareKpis').innerHTML=keys.map(k=>{const t=R.table[k];return `<div class="kpi"><b>${fmt(100*t.quality,0)} % · ${t.modes} modes</b><span>${k} · dispersion ${fmt(t.spread,3)}</span></div>`}).join('')})();
})();

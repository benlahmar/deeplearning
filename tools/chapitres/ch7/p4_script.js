// Chapitre 7 — figures et laboratoires.
(function(){
const {$,fmt,el,text,rng,gauss,softmax,plot}=window.Carnet;
const R=@@RESULTS@@;
const DIG=@@DIGITS@@;
const METH={invariance:['invariance seule','var(--ochre)'],vicreg_sans_variance:['VICReg sans variance','var(--ochre)'],infonce:['InfoNCE','var(--blue)'],vicreg:['VICReg','var(--green)'],byol:['BYOL','var(--ink)'],ijepa:['mini I-JEPA','var(--muted)']};

// ---------- 7.1 · Taxonomie ----------
(()=>{const svg=$('#taxoFig'),defs=el('defs',{},svg),mk=el('marker',{id:'tarr',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto'},defs);el('path',{d:'M0 0L10 5L0 10z',class:'f-m'},mk);
 const A=(d)=>el('path',{d,class:'st-m',fill:'none','stroke-width':1.4,'marker-end':'url(#tarr)'},svg);
 const box=(x,y,w,h,label,cls)=>{el('rect',{x,y,width:w,height:h,rx:6,class:cls},svg);text(svg,x+w/2,y+h/2+1,label,'lbl')};
 const panels=[['Générative',20],['Plongements joints',235],['JEPA',450]];
 panels.forEach(([t,x])=>text(svg,x+85,18,t,'axlabel',{'text-anchor':'middle'}));
 // générative
 box(30,205,50,28,'x','box');A('M55 205V172');box(17,140,76,30,'encodeur','box-a');A('M55 140V112');box(17,80,76,30,'décodeur','box-a');A('M93 95H120');box(122,80,50,30,'ŷ','box-b');box(122,205,50,28,'y','box');A('M147 205V112');text(svg,147,60,'D(ŷ, y)','lbl-m');
 // joint
 box(245,205,50,28,'x','box');box(365,205,50,28,'y','box');A('M270 205V172');A('M390 205V172');box(240,140,60,30,'enc_x','box-a');box(360,140,60,30,'enc_y','box-a');A('M270 140V112');A('M390 140V112');text(svg,270,98,'s_x','lbl-m');text(svg,390,98,'s_y','lbl-m');text(svg,330,72,'D(s_x, s_y)','lbl-m');el('path',{d:'M285 98H375',class:'s-d'},svg);
 // jepa
 box(460,205,50,28,'x','box');box(580,205,50,28,'y','box');A('M485 205V172');A('M605 205V172');box(455,140,60,30,'enc_x','box-a');box(575,140,60,30,'enc_y','box-a');A('M485 140V125');text(svg,485,112,'s_x','lbl-m');A('M500 108Q520 90 528 84');box(520,62,50,26,'préd.','box-b');text(svg,545,46,'z','lbl-m');A('M545 52V60');A('M570 75H590');text(svg,606,78,'ŝ_y','lbl-m');A('M605 140V100');text(svg,598,124,'D(ŝ_y, s_y)','tick',{'text-anchor':'end'});
})();

// ---------- 7.2 · Vues ----------
function augmentJS(img,r,doShift,doErase){let x=img.map(v=>v/16);if(doShift){const dx=Math.floor(r()*3)-1,dy=Math.floor(r()*3)-1;const y=new Array(64).fill(0);for(let i=0;i<8;i++)for(let j=0;j<8;j++){y[((i+dy+8)%8)*8+((j+dx+8)%8)]=x[i*8+j]}x=y}
 if(doErase&&r()<.5){const i0=Math.floor(r()*6),j0=Math.floor(r()*6);for(let i=i0;i<i0+3;i++)for(let j=j0;j<j0+3;j++)x[i*8+j]=0}
 const s=.7+.6*r(),g=gauss(r);return x.map(v=>Math.max(0,Math.min(1,v*s+.05*g())))}
let viewSeed=1;
function drawViews(){const svg=$('#viewsPlot');svg.innerHTML='';const r=rng(viewSeed),c=7,shift=$('#vShift').checked,erase=$('#vErase').checked;
 const draw=(img,x0,y0)=>{img.forEach((v,k)=>el('rect',{x:x0+(k%8)*c,y:y0+Math.floor(k/8)*c,width:c,height:c,fill:'var(--ink)','fill-opacity':v.toFixed(2)},svg));el('rect',{x:x0,y:y0,width:8*c,height:8*c,fill:'none',class:'st-m','stroke-width':.6},svg)};
 ['original','vue 1','vue 2'].forEach((t,row)=>text(svg,4,26+row*72,t,'tick'));
 DIG.forEach((d,i)=>{const x0=62+i*58;draw(d.map(v=>v/16),x0,4);draw(augmentJS(d,r,shift,erase),x0,76);draw(augmentJS(d,r,shift,erase),x0,148)});}
$('#newViews').onclick=()=>{viewSeed++;drawViews()};['vShift','vErase'].forEach(id=>$('#'+id).onchange=drawViews);drawViews();

// ---------- 7.3 · InfoNCE ----------
function drawNce(){const tau=+$('#nTau').value,hard=+$('#nHard').value;$('#nTauOut').textContent=fmt(tau,2);$('#nHardOut').textContent=fmt(hard,2);
 const n=4,M=2*n,r=rng(5),g=gauss(r),labels=[];for(let i=0;i<n;i++){labels.push('x'+(i+1));labels.push('x'+(i+1)+'′')}
 const S=[];for(let i=0;i<M;i++){S.push([]);for(let j=0;j<M;j++){S[i].push(0)}}
 for(let i=0;i<M;i++)for(let j=i;j<M;j++){let v;if(i===j)v=1;else if((i>>1)===(j>>1))v=.85;else v=Math.max(-1,Math.min(.95,hard+.12*g()));S[i][j]=S[j][i]=v}
 const svg=$('#ncePlot');svg.innerHTML='';const c=30,x0=90,y0=40;let loss=0;
 labels.forEach((l,j)=>text(svg,x0+j*c+c/2,y0-8,l,'tick',{'text-anchor':'middle'}));
 for(let i=0;i<M;i++){text(svg,x0-8,y0+i*c+c/2+4,labels[i],'tick',{'text-anchor':'end'});const p=softmax(S[i].map((v,j)=>j===i?-Infinity:v/tau)),pos=i%2?i-1:i+1;loss+=-Math.log(p[pos]);
  for(let j=0;j<M;j++){const rr=el('rect',{x:x0+j*c,y:y0+i*c,width:c,height:c,class:'cell'},svg);if(j===i){rr.setAttribute('fill','var(--line)');rr.setAttribute('fill-opacity','.4')}else{rr.setAttribute('fill','var(--blue)');rr.setAttribute('fill-opacity',(.04+.96*p[j]).toFixed(3));el('title',{},rr).textContent=`${labels[i]} → ${labels[j]} : p = ${fmt(p[j])} (similarité ${fmt(S[i][j])})`}
   if(j===pos)el('rect',{x:x0+j*c+2,y:y0+i*c+2,width:c-4,height:c-4,fill:'none',stroke:'var(--ochre)','stroke-width':2.5,rx:4},svg)}}
 loss/=M;const lx=x0+M*c+30;text(svg,lx,y0+10,'Lecture','axlabel');['ligne i : softmax des similarités','de i avec les 2N − 1 autres','cadre ocre : le positif','diagonale : exclue'].forEach((s,k)=>text(svg,lx,y0+34+k*18,s,'tick'));
 text(svg,lx,y0+130,'InfoNCE = '+fmt(loss,3),'lbl',{'text-anchor':'start'});text(svg,lx,y0+152,'log(2N − 1) = '+fmt(Math.log(M-1),3),'tick');
 $('#nReadout').textContent=`Perte moyenne ${fmt(loss,3)} ; borne d’information mutuelle log(2N − 1) − perte ≈ ${fmt(Math.max(0,Math.log(M-1)-loss),2)} nat.`+(hard>.7?' Des négatifs presque aussi proches que les positifs : la tâche devient difficile.':'');}
['nTau','nHard'].forEach(id=>$('#'+id).oninput=drawNce);drawNce();

// ---------- 7.4 · Spectres ----------
function spectra(svgId,legId){const keys=['aléatoire','pixels','invariance','infonce','vicreg','byol','ijepa'],cols={'aléatoire':['encodeur aléatoire','s-d','var(--muted)'],pixels:['pixels','s-d','var(--muted)'],invariance:['invariance seule','s-b','var(--ochre)'],infonce:['InfoNCE','s-a','var(--blue)'],vicreg:['VICReg','s-c','var(--green)'],byol:['BYOL','s-a','var(--ink)'],ijepa:['mini I-JEPA','s-c','var(--muted)']};
 const series=keys.filter(k=>R.results[k].spectrum.length).map(k=>{const s=R.results[k].spectrum;return {pts:s.map((v,i)=>[i+1,Math.max(v/s[0],1e-4)]),cls:cols[k][1]}});
 const t=plot($('#'+svgId),{w:640,h:280,x:[1,32],y:[1e-3,1],ylog:true,xticks:[1,8,16,24,32],yticks:[1e-3,1e-2,1e-1,1],xlabel:'rang de la valeur singulière',ylabel:'σₖ / σ₁',fmty:v=>v>=1?'1':v.toExponential(0).replace('e-','×10⁻').replace('1×',''),series});
 const svg=$('#'+svgId);series.forEach((s,i)=>{const k=keys.filter(k=>R.results[k].spectrum.length)[i];if(k==='byol'||k==='ijepa'){[...svg.querySelectorAll('path')].slice(-series.length)[i].setAttribute('style','stroke:'+cols[k][2])}});
 $('#'+legId).innerHTML=keys.filter(k=>R.results[k].spectrum.length).map(k=>`<span><i style="background:${cols[k][2]}"></i>${cols[k][0]} · rang ${fmt(R.results[k].rank,1)}</span>`).join('');}
spectra('spectraPlot','spectraLegend');

// ---------- 7.5 · Moyenne mobile ----------
const WALK=(()=>{const g=gauss(rng(21));let v=0;const a=[];for(let t=0;t<300;t++){v+=.15*g()+(t>150?.02:0);a.push(v)}return a})();
function drawEma(){const m=+$('#eM').value;$('#eMOut').textContent=fmt(m,3);let e=WALK[0];const ema=WALK.map(v=>(e=m*e+(1-m)*v));
 const lo=Math.min(...WALK)-.3,hi=Math.max(...WALK)+.3;
 plot($('#emaPlot'),{w:640,h:230,x:[0,300],y:[lo,hi],xticks:[0,100,200,300],yticks:[],xlabel:'mise à jour',ylabel:'un paramètre',series:[{pts:WALK.map((v,i)=>[i,v]),cls:'s-d'},{pts:ema.map((v,i)=>[i,v]),cls:'s-a'}]});
 text($('#emaPlot'),600,30,'mémoire ≈ '+(m>0?fmt(1/(1-m),0):'1')+' mises à jour','tick',{'text-anchor':'end'});}
$('#eM').oninput=drawEma;drawEma();

// ---------- 7.7 · Masques ----------
(()=>{const svg=$('#maskFig'),n=14,c=12,r=rng(4);
 const grid=(x0,title,fill)=>{text(svg,x0+n*c/2,16,title,'axlabel',{'text-anchor':'middle'});for(let i=0;i<n;i++)for(let j=0;j<n;j++){const f=fill(i,j);el('rect',{x:x0+j*c,y:28+i*c,width:c-1,height:c-1,fill:f[0],'fill-opacity':f[1]},svg)}};
 const keep=Array.from({length:n*n},()=>r()<.25);
 grid(40,'MAE : 75 % masqués, cible en pixels',(i,j)=>keep[i*n+j]?['var(--blue)',.75]:['var(--line)',.35]);
 const blocks=[[1,1,4,5],[2,8,5,4],[8,2,4,4],[9,8,4,5]];const inT=(i,j)=>blocks.some(([a,b,h,w])=>i>=a&&i<a+h&&j>=b&&j<b+w);
 grid(380,'I-JEPA : 4 blocs cibles, contexte',(i,j)=>inT(i,j)?['var(--ochre)',.8]:(i>=0&&i<14&&j>=0&&j<13)?['var(--blue)',.55]:['var(--line)',.35]);})();

// ---------- 7.8 · Architecture JEPA ----------
(()=>{const svg=$('#jepaFig'),defs=el('defs',{},svg),mk=el('marker',{id:'jarr',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto'},defs);el('path',{d:'M0 0L10 5L0 10z',class:'f-m'},mk);
 const A=d=>el('path',{d,class:'st-m',fill:'none','stroke-width':1.4,'marker-end':'url(#jarr)'},svg);
 const box=(x,y,w,h,l,cls,s)=>{el('rect',{x,y,width:w,height:h,rx:6,class:cls},svg);text(svg,x+w/2,y+h/2-(s?7:0)+1,l,'lbl');if(s)text(svg,x+w/2,y+h/2+11,s,'lbl-s')};
 const img=(x0,y0,mode)=>{for(let i=0;i<6;i++)for(let j=0;j<6;j++){const t=(i>=1&&i<3&&j>=3&&j<5);el('rect',{x:x0+j*12,y:y0+i*12,width:11,height:11,fill:mode==='ctx'?(t?'var(--line)':'var(--blue)'):(t?'var(--ochre)':'var(--line)'),'fill-opacity':mode==='ctx'?(t?.3:.55):(t?.8:.3)},svg)}};
 img(20,30,'ctx');text(svg,56,22,'contexte x','tick',{'text-anchor':'middle'});img(20,170,'tgt');text(svg,56,252,'image complète y','tick',{'text-anchor':'middle'});
 A('M95 66H138');box(140,44,150,44,'encodeur de contexte','box-a','f_θ, avec gradient');
 A('M95 206H138');box(140,184,150,44,'encodeur cible','box','EMA de f_θ, sans gradient');
 A('M290 66H328');box(330,44,120,44,'prédicteur g_φ','box-b','+ positions z');text(svg,390,30,'z : où prédire ?','tick',{'text-anchor':'middle'});
 A('M450 66H486');text(svg,494,70,'ŝ_y','lbl-m',{'text-anchor':'start'});A('M290 206H486');text(svg,494,210,'s_y (bloc cible)','lbl-m',{'text-anchor':'start'});
 el('path',{d:'M505 82V190',class:'s-d'},svg);text(svg,515,140,'‖ŝ_y − s_y‖²','lbl-m',{'text-anchor':'start'});
 el('path',{d:'M215 88V184',stroke:'var(--muted)','stroke-dasharray':'3 4',fill:'none'},svg);text(svg,222,140,'moyenne mobile','tick');})();

// ---------- Atelier 2 · Rang au fil de l'entraînement ----------
(()=>{const keys=Object.keys(METH),cls={invariance:'s-b',vicreg_sans_variance:'s-b',infonce:'s-a',vicreg:'s-c',byol:'s-a',ijepa:'s-d'};
 plot($('#rankPlot'),{w:640,h:280,x:[0,1400],y:[0,45],xticks:[0,350,700,1050,1400],yticks:[0,10,20,30,40],xlabel:'mise à jour',ylabel:'rang effectif',series:keys.map(k=>({pts:R.histories[k],cls:cls[k],dots:null}))});
 const paths=[...$('#rankPlot').querySelectorAll('path')].slice(-keys.length);keys.forEach((k,i)=>{if(k==='byol')paths[i].setAttribute('style','stroke:var(--ink)');if(k==='vicreg_sans_variance')paths[i].setAttribute('style','stroke-dasharray:6 4')});
 $('#rankLegend').innerHTML=keys.map(k=>`<span><i style="background:${METH[k][1]}"></i>${METH[k][0]}</span>`).join('');})();

// ---------- Atelier 3 · Évaluation ----------
(()=>{const keys=['aléatoire','pixels','invariance','vicreg_sans_variance','infonce','vicreg','byol','ijepa'],names={'aléatoire':'aléatoire',pixels:'pixels',invariance:'invariance',vicreg_sans_variance:'VICReg − var.',infonce:'InfoNCE',vicreg:'VICReg',byol:'BYOL',ijepa:'mini I-JEPA'};
 const svg=$('#evalPlot');svg.setAttribute('viewBox','0 0 640 290');const x0=50,y0=20,h=200,gw=(620-x0)/keys.length,bw=gw/4.2;
 [0,.25,.5,.75,1].forEach(v=>{el('line',{x1:x0,x2:620,y1:y0+h-h*v,y2:y0+h-h*v,class:'gridline'},svg);text(svg,x0-6,y0+h-h*v+4,fmt(100*v,0)+' %','tick',{'text-anchor':'end'})});
 keys.forEach((k,i)=>{const r=R.results[k],gx=x0+i*gw+gw*.1;[[r.probe50,'var(--blue)'],[r.probe_all,'var(--green)'],[r.shift_all,'var(--ochre)']].forEach(([v,c],j)=>{const rr=el('rect',{x:gx+j*bw,y:y0+h-h*v,width:bw-2,height:h*v,fill:c,'fill-opacity':.85},svg);el('title',{},rr).textContent=fmt(100*v,1)+' %'});
  text(svg,gx+1.5*bw,y0+h+16,names[k],'tick',{'text-anchor':'middle'});text(svg,gx+2.5*bw,y0+h-h*r.shift_all-5,fmt(100*r.shift_all,0),'tick',{'text-anchor':'middle'})});
 text(svg,330,282,'Sonde linéaire sur les représentations gelées','axlabel',{'text-anchor':'middle'});})();

// ---------- Atelier 4 · JEPA séquentielle ----------
(()=>{const svg=$('#seqPlot');svg.setAttribute('viewBox','0 0 640 200');const items=[...Object.entries(R.seq).map(([k,v])=>[k,v[0],v[1]]),['GPT autorégressif + sonde (ch. 6)',.978,null]];
 const x0=260,x1=520,sx=v=>x0+(x1-x0)*v;items.forEach(([k,v,rk],i)=>{const y=14+i*44;text(svg,x0-10,y+20,k,'tick',{'text-anchor':'end'});el('rect',{x:x0,y,width:sx(v)-x0,height:28,fill:k.includes('naïf')?'var(--ochre)':k.includes('EMA')?'var(--green)':k.includes('GPT')?'var(--blue)':'var(--muted)','fill-opacity':.85},svg);text(svg,sx(v)+6,y+19,fmt(100*v,1)+' %'+(rk?` · rang ${fmt(rk,1)}`:''),'tick')});
 el('line',{x1:sx(.25),x2:sx(.25),y1:6,y2:186,class:'s-d'},svg);text(svg,sx(.25)+4,196,'hasard 25 %','tick');})();
})();

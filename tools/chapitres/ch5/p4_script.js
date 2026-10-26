// Chapitre 5 — figures et laboratoires.
(function(){
const {$,fmt,el,text,rng,gauss,softmax,plot}=window.Carnet;
const RESULTS=@@RESULTS@@;
const SCALE=@@SCALE@@; // [d_k, Var(q·k), H brut, H normalisé, ‖J‖ brut, ‖J‖ normalisé]
const entropy=p=>-p.reduce((a,x)=>a+(x>0?x*Math.log(x):0),0);

// ---------- 5.2 · Une requête interroge six clés ----------
const KEYS=[['chat',20,1.15],['chien',48,1.0],['tapis',140,1.1],['dort',205,.95],['sur',265,1.05],['le',318,.9]].map(([w,a,r])=>({w,x:r*Math.cos(a*Math.PI/180),y:r*Math.sin(a*Math.PI/180)}));
function drawQuery(){
 const svg=$('#qPlot');svg.innerHTML='';const ang=+$('#qAngle').value*Math.PI/180,beta=+$('#qScale').value;
 $('#qAngleOut').textContent=$('#qAngle').value+'°';$('#qScaleOut').textContent=fmt(beta);
 const q=[Math.cos(ang),Math.sin(ang)],s=KEYS.map(k=>4*beta*(q[0]*k.x+q[1]*k.y)),p=softmax(s);
 const cx=150,cy=150,R=105,X=v=>cx+R*v,Y=v=>cy-R*v;
 el('circle',{cx,cy,r:R,class:'gridline',fill:'none'},svg);el('path',{d:`M${cx-R-12} ${cy}H${cx+R+12}M${cx} ${cy-R-12}V${cy+R+12}`,class:'gridline'},svg);
 const defs=el('defs',{},svg),mk=el('marker',{id:'qarr',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto'},defs);el('path',{d:'M0 0L10 5L0 10z',class:'f-b'},mk);
 KEYS.forEach((k,i)=>{el('circle',{cx:X(k.x),cy:Y(k.y),r:5+16*p[i],class:'f-a','fill-opacity':.25+.6*p[i]},svg);text(svg,X(k.x)+(k.x>0?14:-14),Y(k.y)-10,k.w,'tick',{'text-anchor':k.x>0?'start':'end'})});
 el('line',{x1:cx,y1:cy,x2:X(.7*q[0]),y2:Y(.7*q[1]),class:'st-b','stroke-width':3,'marker-end':'url(#qarr)'},svg);text(svg,X(.82*q[0]),Y(.82*q[1])+4,'q','lbl-m');
 const o=KEYS.reduce((a,k,i)=>[a[0]+p[i]*k.x,a[1]+p[i]*k.y],[0,0]);
 el('path',{d:`M${X(o[0])} ${Y(o[1])-8}l8 8l-8 8l-8 -8z`,class:'f-c'},svg);
 el('path',{d:'M22 282l6 6l-6 6l-6 -6z',class:'f-c'},svg);text(svg,34,292,'sortie Σ αⱼ vⱼ · clés kⱼ (= valeurs vⱼ ici)','tick');
 const bx=345,bw=270,bh=200,by=40,w=bw/KEYS.length;
 el('path',{d:`M${bx} ${by}V${by+bh}H${bx+bw}`,class:'ax'},svg);[0,.5,1].forEach(v=>{el('line',{x1:bx,x2:bx+bw,y1:by+bh-v*bh,y2:by+bh-v*bh,class:'gridline'},svg);text(svg,bx-6,by+bh-v*bh+4,fmt(v,1),'tick',{'text-anchor':'end'})});
 const u=1/KEYS.length;el('line',{x1:bx,x2:bx+bw,y1:by+bh-u*bh,y2:by+bh-u*bh,class:'s-d'},svg);
 KEYS.forEach((k,i)=>{el('rect',{x:bx+i*w+6,y:by+bh-p[i]*bh,width:w-12,height:p[i]*bh,class:'f-a','fill-opacity':.85},svg);text(svg,bx+i*w+w/2,by+bh+16,k.w,'tick',{'text-anchor':'middle'});text(svg,bx+i*w+w/2,by+bh-p[i]*bh-6,fmt(p[i]),'tick',{'text-anchor':'middle'})});
 text(svg,bx+bw/2,by-14,'Poids d’attention αⱼ = softmax(q·kⱼ)','axlabel',{'text-anchor':'middle'});
 const H=entropy(p),best=p.indexOf(Math.max(...p));
 $('#qReadout').textContent=`Entropie : ${fmt(H)} (max ${fmt(Math.log(6))}) · poids maximal ${fmt(p[best])} sur « ${KEYS[best].w} »`+(p[best]>.9?' — recherche quasi exacte, softmax saturé.':H>1.6?' — moyenne presque uniforme.':'.');
}
['qAngle','qScale'].forEach(id=>$('#'+id).oninput=drawQuery);drawQuery();

// ---------- 5.3 · Échelle des scores ----------
plot($('#scaleEntropy'),{w:320,h:240,x:[2,1024],y:[0,3.6],xlog:true,xticks:[2,8,32,128,512],yticks:[0,1,2,3],xlabel:'d_k',ylabel:'Entropie',fmty:v=>String(v),
 series:[{pts:SCALE.map(r=>[r[0],Math.log(32)]),cls:'s-d'},{pts:SCALE.map(r=>[r[0],r[2]]),cls:'s-b',dots:'f-b'},{pts:SCALE.map(r=>[r[0],r[3]]),cls:'s-a',dots:'f-a'}]});
plot($('#scaleJac'),{w:320,h:240,x:[2,1024],y:[0,.35],xlog:true,xticks:[2,8,32,128,512],yticks:[0,.1,.2,.3],xlabel:'d_k',ylabel:'‖∂p/∂s‖ moyenne',fmty:v=>fmt(v,1),
 series:[{pts:SCALE.map(r=>[r[0],r[4]]),cls:'s-b',dots:'f-b'},{pts:SCALE.map(r=>[r[0],r[5]]),cls:'s-a',dots:'f-a'}]});

// ---------- 5.4 · Masques ----------
const TOK=['Le','chat','dort','sur','le','tapis','⟨pad⟩','⟨pad⟩'];
const MS=(()=>{const g=gauss(rng(7)),d=12,E=TOK.map((t,i)=>Array.from({length:d},()=>g()+(t==='le'||t==='Le'?.9:0)));
 const Wq=Array.from({length:d},()=>Array.from({length:d},()=>g()/Math.sqrt(d))),Wk=Array.from({length:d},()=>Array.from({length:d},()=>g()/Math.sqrt(d)));
 const mul=(v,W)=>W[0].map((_,j)=>v.reduce((a,x,i)=>a+x*W[i][j],0));const Q=E.map(v=>mul(v,Wq)),K=E.map(v=>mul(v,Wk));
 return Q.map(q=>K.map(k=>q.reduce((a,x,i)=>a+x*k[i],0)/Math.sqrt(d)*2.2))})();
function drawMask(){
 const svg=$('#maskPlot');svg.innerHTML='';const causal=$('#mCausal').checked,pad=$('#mPad').checked,sc=+$('#mTemp').value;$('#mTempOut').textContent=fmt(sc,1);
 const n=TOK.length,c=31,x0=150,y0=62,valid=TOK.map(t=>!(pad&&t==='⟨pad⟩'));let masked=0,nanRows=0;
 TOK.forEach((t,j)=>text(svg,x0+j*c+c/2,y0-8,t,'tick',{'text-anchor':'start',transform:`rotate(-45 ${x0+j*c+c/2} ${y0-8})`}));
 text(svg,x0+n*c/2,16,'clés (colonnes)','axlabel',{'text-anchor':'middle'});text(svg,22,y0+n*c/2,'requêtes','axlabel',{'text-anchor':'middle',transform:`rotate(-90 22 ${y0+n*c/2})`});
 TOK.forEach((t,i)=>{
  text(svg,x0-8,y0+i*c+c/2+4,t,'tick',{'text-anchor':'end'});
  const allow=TOK.map((_,j)=>(!causal||j<=i)&&valid[j]&&valid[i]);masked+=allow.filter(a=>!a).length;
  const s=MS[i].map((v,j)=>allow[j]?v*sc:-Infinity),any=allow.some(Boolean),p=any?softmax(s):s.map(()=>0);if(!any)nanRows++;
  p.forEach((v,j)=>{const r=el('rect',{x:x0+j*c,y:y0+i*c,width:c,height:c,class:'cell'},svg);
   if(allow[j]){r.setAttribute('fill','var(--blue)');r.setAttribute('fill-opacity',(.06+.94*v).toFixed(3));const tt=el('title',{},r);tt.textContent=`${t} → ${TOK[j]} : ${fmt(v)}`;if(v>.3)text(svg,x0+j*c+c/2,y0+i*c+c/2+4,fmt(v,1),'tick',{'text-anchor':'middle',fill:'var(--surface)'})}
   else{r.setAttribute('fill','var(--line)');r.setAttribute('fill-opacity','.35');text(svg,x0+j*c+c/2,y0+i*c+c/2+4,'×','tick',{'text-anchor':'middle'})}});
  text(svg,x0+n*c+10,y0+i*c+c/2+4,any?'Σ = 1':'0/0 → 0','tick',{fill:any?'var(--muted)':'var(--ochre)'});
 });
 const lx=x0+n*c+80;text(svg,lx,y0+10,'Lecture','axlabel');
 ['Ligne i : distribution','de la requête i sur','les clés autorisées.','× : poids forcé à 0'].forEach((s,k)=>text(svg,lx,y0+34+k*18,s,'tick'));
 $('#mReadout').textContent=`${masked} poids masqués sur ${n*n}.`+(causal?' Matrice triangulaire : la première ligne n’a qu’une clé autorisée.':'')+(nanRows?` ${nanRows} lignes entièrement masquées : sorties forcées à 0 et exclues de la perte.`:'')+(pad?'':' Sans masque de padding, des requêtes réelles lisent les jetons de remplissage, qui n’ont aucun sens.');
}
['mCausal','mPad'].forEach(id=>$('#'+id).onchange=drawMask);$('#mTemp').oninput=drawMask;drawMask();

// ---------- 5.5 · Têtes illustratives ----------
(()=>{const g=$('#mhaHeads'),n=6,c=10,pats=[(i,j)=>j===i-1||(i===0&&j===0)?1:.04,(i,j)=>j===0?1:.05,(i,j)=>i===j?1:.06,(i,j)=>((i*3+j*5)%7<2)?.8:.08],names=[['tête 1','précédent'],['tête 2','début'],['tête 3','soi-même'],['tête 4','contenu']];
 const xs=[0,1,2,3].map(r=>125+r*85),y=62;
 el('path',{d:`M85 100H100V40H${xs[3]+30}`,class:'gridline',fill:'none','stroke-width':1.5},g);text(g,270,32,'Q, K, V découpés en h = 4 sous-espaces de taille d/4','tick',{'text-anchor':'middle'});
 el('path',{d:`M${xs[0]+30} 150V170H505V133`,class:'arrow'},g);
 pats.forEach((f,r)=>{const x=xs[r];el('path',{d:`M${x+30} 40V${y-3}`,class:'arrow'},g);el('path',{d:`M${x+30} ${y+n*c+30}V170`,class:'gridline'},g);
  for(let i=0;i<n;i++)for(let j=0;j<n;j++)el('rect',{x:x+j*c,y:y+i*c,width:c,height:c,class:'cell',fill:'var(--blue)','fill-opacity':f(i,j)},g);
  text(g,x+30,y+n*c+14,names[r][0],'tick',{'text-anchor':'middle'});text(g,x+30,y+n*c+27,names[r][1],'tick',{'text-anchor':'middle'})});})();

// ---------- 5.6 · Encodages de position ----------
(()=>{const svg=$('#peHeat'),P=64,D=64,cw=4,ch=3.3,x0=40,y0=20;svg.setAttribute('viewBox','0 0 320 270');
 for(let p=0;p<P;p++)for(let i=0;i<D;i++){const w=Math.pow(10000,-2*Math.floor(i/2)/D),v=i%2?Math.cos(p*w):Math.sin(p*w);el('rect',{x:x0+i*cw,y:y0+p*ch,width:cw,height:ch,fill:v>=0?'var(--blue)':'var(--ochre)','fill-opacity':Math.abs(v).toFixed(2)},svg)}
 text(svg,x0+D*cw/2,y0+P*ch+20,'dimension 0 → 63','axlabel',{'text-anchor':'middle'});text(svg,16,y0+P*ch/2,'position 0 → 63','axlabel',{'text-anchor':'middle',transform:`rotate(-90 16 ${y0+P*ch/2})`});
 const pts=[];for(let k=0;k<=64;k++){let s=0;for(let i=0;i<D/2;i++)s+=Math.cos(k*Math.pow(10000,-2*i/D));pts.push([k,s/(D/2)])}
 plot($('#peSim'),{w:320,h:270,x:[0,64],y:[-.2,1],xticks:[0,16,32,48,64],yticks:[0,.5,1],xlabel:'décalage k',ylabel:'similarité',fmty:v=>fmt(v,1),series:[{pts,cls:'s-a'}]})})();

// ---------- 5.6 · RoPE ----------
function drawRope(){
 const svg=$('#ropePlot');svg.innerHTML='';const m=+$('#rM').value,n=+$('#rN').value,th=.3;$('#rMOut').textContent=m;$('#rNOut').textContent=n;
 const qa=20*Math.PI/180+m*th,ka=70*Math.PI/180+n*th,qn=1,kn=.85,dot=qn*kn*Math.cos(ka-qa);
 const cx=140,cy=140,R=100,defs=el('defs',{},svg);[['ra','f-b'],['rb','f-c']].forEach(([id,c])=>{const mk=el('marker',{id,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:7,markerHeight:7,orient:'auto'},defs);el('path',{d:'M0 0L10 5L0 10z',class:c},mk)});
 el('circle',{cx,cy,r:R,class:'gridline',fill:'none'},svg);
 el('line',{x1:cx,y1:cy,x2:cx+R*qn*Math.cos(qa),y2:cy-R*qn*Math.sin(qa),class:'st-b','stroke-width':3,'marker-end':'url(#ra)'},svg);
 el('line',{x1:cx,y1:cy,x2:cx+R*kn*Math.cos(ka),y2:cy-R*kn*Math.sin(ka),class:'st-c','stroke-width':3,'marker-end':'url(#rb)'},svg);
 text(svg,cx+(R+16)*Math.cos(qa),cy-(R+16)*Math.sin(qa)+4,'R(mθ)q','tick',{'text-anchor':'middle',fill:'var(--ochre)'});text(svg,cx+(R*kn+18)*Math.cos(ka),cy-(R*kn+18)*Math.sin(ka)+4,'R(nθ)k','tick',{'text-anchor':'middle',fill:'var(--green)'});
 text(svg,cx,cy+R+30,'Une paire de coordonnées, fréquence θ = 0,3','axlabel',{'text-anchor':'middle'});
 const pts=[];for(let d=-20;d<=20;d++)pts.push([d,qn*kn*Math.cos(50*Math.PI/180+d*th)]);
 const g=el('g',{transform:'translate(300 0)'},svg),inner=el('svg',{x:0,y:0,width:330,height:270},g);
 const t=plot(inner,{w:330,h:270,x:[-20,20],y:[-1,1],xticks:[-20,-10,0,10,20],yticks:[-1,0,1],xlabel:'écart n − m',ylabel:'score',series:[{pts,cls:'s-a'}]});
 el('circle',{cx:t.tx(n-m),cy:t.ty(dot),r:6,class:'f-b'},inner);
 $('#rReadout').textContent=`Score = ${fmt(dot,3)} · écart n − m = ${n-m}. Le point se déplace sur la courbe seulement si l’écart change.`;
}
['rM','rN'].forEach(id=>$('#'+id).oninput=drawRope);$('#rShift').onclick=()=>{const m=+$('#rM').value,n=+$('#rN').value;if(Math.max(m,n)+3>20){$('#rM').value=Math.max(0,m-17);$('#rN').value=Math.max(0,n-17)}else{$('#rM').value=m+3;$('#rN').value=n+3}drawRope()};drawRope();

// ---------- 5.8 · Coûts ----------
const bytes=b=>{const u=['o','Kio','Mio','Gio','Tio','Pio'];let i=0;while(b>=1024&&i<u.length-1){b/=1024;i++}return fmt(b,b<10?2:1)+' '+u[i]};
function drawCost(){
 const n=Math.round(Math.pow(2,+$('#cN').value)),d=+$('#cD').value,L=+$('#cL').value,kv=+$('#cKV').value,h=32,dh=d/h;$('#cNOut').textContent=n.toLocaleString('fr-FR')+' jetons';
 const xs=[];for(let e=7;e<=18;e+=.25)xs.push(Math.pow(2,e));
 const t=plot($('#costPlot'),{w:640,h:300,x:[128,262144],y:[1e8,1e18],xlog:true,ylog:true,xticks:[128,1024,8192,65536,262144],yticks:[1e8,1e10,1e12,1e14,1e16,1e18],xlabel:'longueur de séquence n',ylabel:'FLOPs par couche',fmtx:v=>v>=1024?(v/1024)+' k':String(v),fmty:v=>'1e'+Math.round(Math.log10(v)),
  series:[{pts:xs.map(x=>[x,24*x*d*d]),cls:'s-c'},{pts:xs.map(x=>[x,4*x*x*d]),cls:'s-b'}]});
 const svg=$('#costPlot');el('line',{x1:t.tx(n),x2:t.tx(n),y1:t.m.t,y2:t.H-t.m.b,class:'s-d'},svg);
 if(6*d>=128&&6*d<=262144){el('circle',{cx:t.tx(6*d),cy:t.ty(24*6*d*d*d),r:5,class:'f-ink'},svg);text(svg,t.tx(6*d)+8,t.ty(24*6*d*d*d)+18,'n = 6d','tick')}
 const att=4*n*n*d,dense=24*n*d*d,share=att/(att+dense);
 const k=[[fmt(100*share,1)+' %','part de l’attention dans les FLOPs'],[fmt(L*(att+dense)/1e12,1)+' TFLOP','passage avant, toute la pile'],[bytes(h*n*n*2),'matrice A naïve, une couche'],[bytes(2*L*kv*dh*n*2),'cache KV (16 bits)']];
 $('#cKpis').innerHTML=k.map(([v,l])=>`<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join('');
}
['cN'].forEach(id=>$('#'+id).oninput=drawCost);['cD','cL','cKV'].forEach(id=>$('#'+id).onchange=drawCost);drawCost();

// ---------- 5.8 · Blocs de FlashAttention ----------
(()=>{const g=$('#flashBlocks');for(let b=0;b<5;b++){const x=130+b*64,gr=el('g',{class:'math-highlight',style:`animation-delay:${b}s;animation-duration:5s`},g);el('rect',{x,y:40,width:52,height:52,rx:5,class:'box-a'},gr);el('rect',{x,y:110,width:52,height:52,rx:5,class:'box-c'},gr);text(gr,x+26,67,'K'+(b+1),'lbl-m');text(gr,x+26,137,'V'+(b+1),'lbl-m')}
 el('path',{d:'M90 105H126M448 105H460',class:'arrow'},g)})();

// ---------- Atelier 3 · Résultats réels ----------
const NAMES={none:['aucune','s-b','var(--ochre)'],sinusoidal:['sinusoïdale','s-c','var(--green)'],learned:['apprise','s-a','var(--blue)'],rope:['RoPE','s-d','var(--muted)']};
plot($('#trainPlot'),{w:640,h:280,x:[0,1450],y:[0,2.5],xticks:[0,250,500,750,1000,1250],yticks:[0,.5,1,1.5,2,2.5],xlabel:'mise à jour',ylabel:'entropie croisée',fmty:v=>fmt(v,1),
 series:Object.keys(NAMES).map(k=>({pts:RESULTS.hist[k],cls:NAMES[k][1]}))});
$('#trainLegend').innerHTML=Object.keys(NAMES).map(k=>`<span><i style="background:${NAMES[k][2]}"></i>${NAMES[k][0]}</span>`).join('');
$('#trainKpis').innerHTML=Object.keys(NAMES).map(k=>`<div class="kpi"><b>${fmt(100*RESULTS.acc[k],1)} %</b><span>accuracy · ${NAMES[k][0]}</span></div>`).join('')+`<div class="kpi"><b>31,7 %</b><span>optimum sans position (TD ex. 3)</span></div>`;
const INPUT=['3','1','4','1','5','9','2','6'];
[['attNone','none','sans position'],['attLearned','learned','positions apprises'],['attRope','rope','RoPE']].forEach(([id,k,title])=>{
 const svg=$('#'+id),c=24,x0=26,y0=34;svg.setAttribute('viewBox','0 0 230 250');text(svg,x0+4*c,16,title,'axlabel',{'text-anchor':'middle'});
 RESULTS.att[k].forEach((row,i)=>{text(svg,x0-8,y0+i*c+c/2+4,INPUT[i],'tick',{'text-anchor':'end'});row.forEach((v,j)=>{const r=el('rect',{x:x0+j*c,y:y0+i*c,width:c,height:c,class:'cell',fill:'var(--blue)','fill-opacity':(.04+.96*v).toFixed(3)},svg);el('title',{},r).textContent=`requête ${i} → clé ${j} : ${fmt(v)}`})});
 INPUT.forEach((t,j)=>text(svg,x0+j*c+c/2,y0+8*c+16,t,'tick',{'text-anchor':'middle'}));
});
})();

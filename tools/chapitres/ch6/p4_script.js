// Chapitre 6 — figures et laboratoires.
(function(){
const {$,fmt,el,text,rng,gauss,softmax,plot}=window.Carnet;
const R=@@RESULTS@@;
const NAMES={scratch:['à partir de zéro','var(--ochre)'],probe:['sonde linéaire','var(--green)'],full:['ajustement complet','var(--blue)'],'lora-1':['LoRA r = 1','var(--muted)'],'lora-4':['LoRA r = 4','var(--muted)'],'lora-16':['LoRA r = 16','var(--muted)']};
const sup=n=>String(n).split('').map(c=>'⁰¹²³⁴⁵⁶⁷⁸⁹'[c]||'⁻').join('');
const sci=v=>{const e=Math.floor(Math.log10(v)+1e-9),m=v/Math.pow(10,e);return fmt(m,1)+' × 10'+sup(e)};
const big=v=>v>=1e12?fmt(v/1e12,1)+' billions':v>=1e9?fmt(v/1e9,1)+' milliards':v>=1e6?fmt(v/1e6,1)+' M':fmt(v,0);
const bytes=b=>{const u=['o','Kio','Mio','Gio','Tio'];let i=0;while(b>=1024&&i<u.length-1){b/=1024;i++}return fmt(b,b<10?2:1)+' '+u[i]};

// ---------- 6.2 · Objectifs ----------
(()=>{const svg=$('#objPlot'),W=['Le','chat','dort','sur','le','tapis','rouge'],cw=58,x0=232;
 const rows=[['Causal (GPT)',W.map((w,i)=>i<6?'v':'-'),'chaque position prédit la suivante'],['Masqué (BERT)',W.map((w,i)=>i===1||i===5?'m':'v'),'[MASK] à 15 % des positions'],['Segments (T5)',W.map((w,i)=>i===2||i===3?'s':'v'),'segment → sentinelle ⟨X⟩, régénéré']];
 rows.forEach(([name,kind,note],r)=>{const y=20+r*78;text(svg,10,y+26,name,'axlabel');text(svg,10,y+44,note,'tick');
  W.forEach((w,i)=>{const x=x0+i*cw,k=kind[i];const cls=k==='v'?'box-a':k==='-'?'box':'box-b';el('rect',{x,y:y+10,width:cw-8,height:30,rx:5,class:cls},svg);
   const lab=k==='m'?'[MASK]':k==='s'?(i===2?'⟨X⟩':''):w;text(svg,x+(cw-8)/2,y+29,lab,'lbl',{});
   if(r===0&&i<6)el('path',{d:`M${x+cw-14} ${y+50}Q${x+cw-4} ${y+64} ${x+cw+18} ${y+42}`,class:'arrow'},svg);
   if(k==='m'||(k==='s'&&i===2))text(svg,x+(cw-8)/2,y+58,k==='m'?'→ '+w:'→ dort sur','tick',{'text-anchor':'middle',fill:'var(--ochre)'})});});
})();

// ---------- 6.3 · Lois d'échelle ----------
const CH={E:1.69,A:406.4,B:410.7,a:.34,b:.28};
const Lnd=(N,D)=>CH.E+CH.A/Math.pow(N,CH.a)+CH.B/Math.pow(D,CH.b);
function optN(C){const G=Math.pow(CH.a*CH.A/(CH.b*CH.B),1/(CH.a+CH.b));return G*Math.pow(C/6,CH.b/(CH.a+CH.b))}
function drawScaling(){
 const e=+$('#sC').value,C=Math.pow(10,e);$('#sCOut').textContent=sci(C)+' FLOPs';
 const Ns=[];for(let x=6;x<=13;x+=.05)Ns.push(Math.pow(10,x));
 const budgets=[e-2,e-1,e,e+1].filter(v=>v>=17.5&&v<=26);
 const series=budgets.map(b=>({pts:Ns.map(N=>[N,Lnd(N,Math.pow(10,b)/(6*N))]).filter(p=>p[1]<4.15),cls:b===e?'s-a':'s-d'}));
 const t=plot($('#scalingPlot'),{w:640,h:300,x:[1e6,1e13],y:[1.8,4.2],xlog:true,xticks:[1e6,1e8,1e10,1e12],yticks:[2,2.5,3,3.5,4],xlabel:'paramètres N (le reste du budget va aux données : D = C / 6N)',ylabel:'perte prédite',fmtx:v=>'10'+sup(Math.round(Math.log10(v))),fmty:v=>fmt(v,1),series});
 budgets.forEach(b=>{const n=optN(Math.pow(10,b)),l=Lnd(n,Math.pow(10,b)/(6*n));el('circle',{cx:t.tx(n),cy:t.ty(l),r:b===e?6:3.5,class:b===e?'f-b':'f-m'},$('#scalingPlot'));if(b!==e)text($('#scalingPlot'),t.tx(n)+6,t.ty(l)-6,sci(Math.pow(10,b)),'tick')});
 const N=optN(C),D=C/(6*N);
 $('#sKpis').innerHTML=[[big(N),'paramètres optimaux N*'],[big(D),'jetons optimaux D*'],[fmt(D/N,0),'jetons par paramètre'],[fmt(Lnd(N,D),3),'perte prédite']].map(([v,l])=>`<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join('');
}
$('#sC').oninput=drawScaling;drawScaling();

// ---------- 6.5 · Figure LoRA ----------
(()=>{const svg=$('#loraFig'),g=gauss(rng(3)),cell=9;
 const mat=(x,y,rows,cols,cls,op)=>{for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){el('rect',{x:x+j*cell,y:y+i*cell,width:cell,height:cell,class:'cell',fill:cls,'fill-opacity':(op?op:.15+.6*Math.abs(Math.tanh(g()))).toFixed(2)},svg)}};
 mat(20,25,16,16,'var(--blue)');text(svg,92,185,'W₀ ∈ ℝ^(d×k)','lbl-m');text(svg,92,205,'gelée','tick',{'text-anchor':'middle'});
 el('path',{d:'M78 10h28v-0M84 10v-6a8 8 0 0 1 16 0v6',class:'st-m',fill:'none','stroke-width':2},svg);el('rect',{x:80,y:9,width:24,height:14,rx:2,class:'f-m'},svg);
 text(svg,200,100,'+','lbl-m',{'font-size':'28'});text(svg,250,100,'α/r','lbl-m');
 mat(285,25,16,2,'var(--ochre)');text(svg,294,185,'B','lbl-m');text(svg,294,205,'d × r, init. 0','tick',{'text-anchor':'middle'});
 mat(325,25,2,16,'var(--ochre)');text(svg,397,60,'A','lbl-m');text(svg,397,78,'r × k, init. aléatoire','tick',{'text-anchor':'middle'});
 text(svg,500,95,'=  W, rang(W − W₀) ≤ r','lbl-m');text(svg,520,130,'entraînés : r(d + k)','tick',{'text-anchor':'middle'});text(svg,520,148,'au lieu de d·k','tick',{'text-anchor':'middle'})})();

// ---------- 6.5 · Calculateur LoRA ----------
function drawLora(){
 const d=+$('#lD').value,L=+$('#lL').value,r=Math.pow(2,+$('#lR').value);$('#lROut').textContent='r = '+r;
 const ff=Math.round(8*d/3),attn=$('#lAttn').checked,ffn=$('#lFfn').checked;
 const total=L*(4*d*d+3*d*ff);let lora=0;if(attn)lora+=L*4*r*(d+d);if(ffn)lora+=L*3*r*(d+ff);
 $('#lKpis').innerHTML=[[big(total),'paramètres des blocs'],[big(lora),'paramètres LoRA'],[fmt(100*lora/total,lora/total<.001?3:2)+' %','part entraînée'],[bytes(lora*8),'état d’Adam (m, v)'],[bytes(total*8),'état d’Adam, ajustement complet']].map(([v,l])=>`<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join('');
}
['lD','lL','lAttn','lFfn'].forEach(id=>$('#'+id).onchange=drawLora);$('#lR').oninput=drawLora;drawLora();

// ---------- 6.6 · Mémoire ----------
const SIZES=[1e9,3e9,7e9,13e9,34e9,70e9];
const PARTS=[['poids','var(--blue)'],['gradients','var(--ochre)'],['copie fp32','var(--green)'],['moments Adam','var(--muted)']];
$('#memLegend').innerHTML=PARTS.map(([n,c])=>`<span><i style="background:${c}"></i>${n}</span>`).join('');
function drawMem(){
 const N=SIZES[+$('#mN').value],f=+$('#mFrac').value;$('#mNOut').textContent=big(N)+' de paramètres';
 const meth=[['Ajustement complet',[2*N,2*N,4*N,8*N]],['LoRA, base 16 bits',[2*N+2*f*N,2*f*N,4*f*N,8*f*N]],['QLoRA, base 4 bits',[.5*N+2*f*N,2*f*N,4*f*N,8*f*N]]];
 const svg=$('#memPlot');svg.innerHTML='';const x0=170,x1=560,max=Math.max(80e9,...meth.map(m=>m[1].reduce((a,b)=>a+b)))*1.05,sx=v=>x0+(x1-x0)*v/max;
 [24e9,48e9,80e9].forEach(v=>{el('line',{x1:sx(v),x2:sx(v),y1:20,y2:205,class:'s-d'},svg);text(svg,sx(v),222,fmt(v/1e9,0)+' Go','tick',{'text-anchor':'middle'})});
 meth.forEach(([name,parts],i)=>{const y=35+i*60;let acc=0;text(svg,x0-10,y+20,name,'tick',{'text-anchor':'end'});
  parts.forEach((v,j)=>{el('rect',{x:sx(acc),y,width:Math.max(0,sx(acc+v)-sx(acc)),height:32,fill:PARTS[j][1],'fill-opacity':.85},svg);acc+=v});
  text(svg,sx(acc)+8,y+21,fmt(acc/1e9,1)+' Go','tick')});
 text(svg,(x0+x1)/2,250,'mémoire hors activations (Go)','axlabel',{'text-anchor':'middle'});
}
$('#mN').oninput=drawMem;$('#mFrac').onchange=drawMem;drawMem();

// ---------- 6.7 · Formats flottants ----------
(()=>{const svg=$('#floatFig'),fmts=[['fp32',1,8,23],['fp16',1,5,10],['bf16',1,8,7]],u=17;
 fmts.forEach(([n,s,e,m],r)=>{const y=12+r*50;text(svg,10,y+20,n,'axlabel');let x=70;
  [[s,'var(--ink)','s'],[e,'var(--ochre)','exposant'],[m,'var(--blue)','mantisse']].forEach(([k,c,lab])=>{for(let i=0;i<k;i++)el('rect',{x:x+i*u,y,width:u-2,height:26,fill:c,'fill-opacity':.75,rx:2},svg);if(k>1)text(svg,x+k*u/2,y+40,lab+' ('+k+')','tick',{'text-anchor':'middle'});x+=k*u+4});});})();

// ---------- 6.7 · Quantification ----------
const QW=(()=>{const g=gauss(rng(11));return Array.from({length:4096},()=>.02*g())})();
function quant(w,bits,group){const q=Math.pow(2,bits-1)-1,out=new Array(w.length),G=group||w.length;for(let s=0;s<w.length;s+=G){let m=0;for(let i=s;i<Math.min(s+G,w.length);i++)m=Math.max(m,Math.abs(w[i]));const d=Math.max(m,1e-12)/q;for(let i=s;i<Math.min(s+G,w.length);i++)out[i]=d*Math.max(-q,Math.min(q,Math.round(w[i]/d)))}return out}
function drawQuant(){
 const bits=+$('#qBits').value,out=$('#qOutlier').checked,grp=$('#qGroup').checked;$('#qBitsOut').textContent=bits+' bits';
 const w=QW.slice();if(out)w[777]=2.0;const wq=quant(w,bits,grp?32:null);
 let num=0,den=0,zeros=0;w.forEach((v,i)=>{num+=(v-wq[i])**2;den+=v*v;if(wq[i]===0)zeros++});
 const svg=$('#quantPlot');svg.innerHTML='';const x0=40,x1=620,lo=-.09,hi=.09,sx=v=>x0+(x1-x0)*(v-lo)/(hi-lo),nb=72,h=new Array(nb).fill(0),hq=new Array(nb).fill(0);
 w.forEach((v,i)=>{const b=Math.floor((v-lo)/(hi-lo)*nb);if(b>=0&&b<nb)h[b]++;const c=Math.floor((wq[i]-lo)/(hi-lo)*nb);if(c>=0&&c<nb)hq[c]++});
 const mx=Math.max(...h),sy=v=>200-170*Math.min(1,v/mx);
 h.forEach((v,b)=>el('rect',{x:x0+b*(x1-x0)/nb,y:sy(v),width:(x1-x0)/nb-1,height:200-sy(v),fill:'var(--blue)','fill-opacity':.45},svg));
 hq.forEach((v,b)=>{if(v)el('rect',{x:x0+b*(x1-x0)/nb+1,y:sy(v),width:(x1-x0)/nb-3,height:200-sy(v),fill:'var(--ochre)','fill-opacity':.8},svg)});
 if(!grp){const q=Math.pow(2,bits-1)-1,d=Math.max(...w.map(Math.abs))/q;for(let k=-q;k<=q;k++){const v=k*d;if(v>=lo&&v<=hi)el('line',{x1:sx(v),x2:sx(v),y1:25,y2:200,class:'s-d'},svg)}}
 el('path',{d:`M${x0} 200H${x1}`,class:'ax'},svg);[-.08,-.04,0,.04,.08].forEach(v=>text(svg,sx(v),216,fmt(v,2),'tick',{'text-anchor':'middle'}));
 text(svg,330,244,'valeur du poids (bleu : original ; ocre : quantifié ; pointillés : niveaux)','axlabel',{'text-anchor':'middle'});
 $('#qReadout').textContent=`Erreur relative ${fmt(100*Math.sqrt(num/den),1)} % · ${fmt(100*zeros/w.length,1)} % des poids arrondis à zéro`+(out&&!grp?' — la valeur aberrante (2,0, hors cadre) fixe le pas de tout le tenseur.':'');
}
$('#qBits').oninput=drawQuant;['qOutlier','qGroup'].forEach(id=>$('#'+id).onchange=drawQuant);drawQuant();

// ---------- 6.8 · Température ----------
const TCLS=['chat','tigre','lynx','chien','renard','voiture','camion','avion'],TLOG=[9,5.5,5,3,2.6,-1,-1.5,-2];
function drawTemp(){const T=+$('#tT').value;$('#tTOut').textContent=fmt(T,1);const p=softmax(TLOG.map(z=>z/T));
 const svg=$('#tempPlot');svg.innerHTML='';const x0=50,bw=70,base=185;
 [0,.5,1].forEach(v=>{el('line',{x1:x0,x2:x0+bw*8,y1:base-160*v,y2:base-160*v,class:'gridline'},svg);text(svg,x0-6,base-160*v+4,fmt(v,1),'tick',{'text-anchor':'end'})});
 p.forEach((v,i)=>{el('rect',{x:x0+i*bw+10,y:base-160*v,width:bw-20,height:160*v,fill:i<3?'var(--blue)':i<5?'var(--green)':'var(--ochre)','fill-opacity':.85},svg);text(svg,x0+i*bw+bw/2,base+16,TCLS[i],'tick',{'text-anchor':'middle'});text(svg,x0+i*bw+bw/2,base-160*v-6,fmt(100*v,v<.1?1:0)+' %','tick',{'text-anchor':'middle'})});
 text(svg,x0+4*bw,222,'Probabilités du maître pour une image de chat : félins (bleu), autres animaux (vert), véhicules (ocre)','axlabel',{'text-anchor':'middle'});}
$('#tT').oninput=drawTemp;drawTemp();

// ---------- Atelier 1 · Préentraînement ----------
(()=>{const t=plot($('#preCurve'),{w:640,h:280,x:[0,2450],y:[1.3,3.3],xticks:[0,500,1000,1500,2000],yticks:[1.5,2,2.5,3],xlabel:'mise à jour',ylabel:'entropie croisée (nat)',fmty:v=>fmt(v,1),
 series:[{pts:[[0,1.86],[2450,1.86]],cls:'s-b'},{pts:[[0,1.52],[2450,1.52]],cls:'s-c'},{pts:[[0,1.475],[2450,1.475]],cls:'s-d'},{pts:R.pre,cls:'s-a'}]})})();

// ---------- Atelier 2 · Stratégies et oubli ----------
(()=>{const svg=$('#methodsPlot'),keys=Object.keys(NAMES);
 const t=plot(svg,{w:640,h:300,x:[100,300000],y:[.4,1],xlog:true,xticks:[100,1000,10000,100000],yticks:[.4,.5,.6,.7,.8,.9,1],xlabel:'paramètres entraînés (échelle log)',ylabel:'accuracy test',fmtx:v=>v.toLocaleString('fr-FR'),fmty:v=>fmt(100*v,0)+' %',series:[]});
 keys.forEach(k=>{const m=R.methods[k],a=m.acc,mu=a.reduce((x,y)=>x+y)/a.length,sd=Math.sqrt(a.reduce((x,y)=>x+(y-mu)**2,0)/a.length),x=t.tx(m.params)+(k==='scratch'?-8:k==='full'?8:0);
  el('line',{x1:x,x2:x,y1:t.ty(mu-sd),y2:t.ty(mu+sd),stroke:NAMES[k][1],'stroke-width':2},svg);el('circle',{cx:x,cy:t.ty(mu),r:6,fill:NAMES[k][1]},svg);
  const o={scratch:[-10,-10,'end'],probe:[10,-10,'start'],full:[-10,-14,'end'],'lora-1':[0,24,'middle'],'lora-4':[0,-14,'middle'],'lora-16':[0,24,'middle']}[k];text(svg,x+o[0],t.ty(mu)+o[1],NAMES[k][0]+' · '+fmt(100*mu,1)+' %','tick',{'text-anchor':o[2]})});})();
(()=>{const svg=$('#forgetPlot');svg.setAttribute('viewBox','0 0 640 240');const f=R.forgetting,keys=Object.keys(f),labels={'préentraîné':'préentraîné','probe':'sonde linéaire','full':'ajustement complet','lora-4':'LoRA r = 4, actif','lora-4, adaptateurs retirés':'LoRA r = 4, B·A retiré'};
 const x0=210,x1=600,max=3.6,sx=v=>x0+(x1-x0)*v/max;
 keys.forEach((k,i)=>{const y=12+i*40,v=f[k];text(svg,x0-10,y+19,labels[k]||k,'tick',{'text-anchor':'end'});el('rect',{x:x0,y,width:sx(v)-x0,height:26,fill:k==='full'||k==='lora-4'?'var(--ochre)':'var(--blue)','fill-opacity':.8},svg);text(svg,sx(v)+6,y+18,fmt(v,3),'tick')});
 el('line',{x1:sx(Math.log(24)),x2:sx(Math.log(24)),y1:8,y2:210,class:'s-d'},svg);text(svg,sx(Math.log(24)),226,'uniforme : log 24','tick',{'text-anchor':'middle'});})();

// ---------- Atelier 3 · Efficacité en étiquettes ----------
(()=>{const cols={scratch:['à partir de zéro','s-b','f-b','var(--ochre)'],probe:['sonde linéaire','s-c','f-c','var(--green)'],full:['ajustement complet','s-a','f-a','var(--blue)'],'lora-4':['LoRA r = 4','s-d','f-m','var(--muted)']};
 plot($('#curvePlot'),{w:640,h:280,x:[12,1400],y:[.4,1],xlog:true,xticks:[16,64,256,1024],yticks:[.4,.5,.6,.7,.8,.9,1],xlabel:'exemples étiquetés',ylabel:'accuracy test',fmty:v=>fmt(100*v,0)+' %',series:Object.keys(cols).map(k=>({pts:R.curve[k],cls:cols[k][1],dots:cols[k][2]}))});
 $('#curveLegend').innerHTML=Object.keys(cols).map(k=>`<span><i style="background:${cols[k][3]}"></i>${cols[k][0]}</span>`).join('');})();
})();

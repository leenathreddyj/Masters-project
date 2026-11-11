/* ================= working system ================= */
var NS='http://www.w3.org/2000/svg';
function seg(id,cb){var g=document.getElementById(id);[].forEach.call(g.querySelectorAll('button'),function(b){b.addEventListener('click',function(){[].forEach.call(g.querySelectorAll('button'),function(x){x.setAttribute('aria-pressed',x===b)});cb(b.dataset.v)})})}
function se(t,a,p){var n=document.createElementNS(NS,t); for(var k in a)n.setAttribute(k,a[k]); p.appendChild(n); return n}
var ENG=engineFactory(), MISS=[1,2,3,4,5], N=ROWS.length;
var ALL=ROWS.map(function(_,i){return i});
var PREPALL=ENG.fitPrep(ROWS,ALL);
var SPLIT=ENG.stratSplit(ROWS,0.2,42);
var FN=[{n:'Pregnancies',s:'Preg',u:'',st:1},{n:'Glucose',s:'Glucose',u:'mg/dL',st:1},{n:'Blood pressure',s:'BP',u:'mm Hg',st:1},{n:'Skin thickness',s:'Skin',u:'mm',st:1},{n:'Insulin',s:'Insulin',u:'µU/mL',st:1},{n:'BMI',s:'BMI',u:'kg/m²',st:.1},{n:'Pedigree function',s:'Pedigree',u:'',st:.01},{n:'Age',s:'Age',u:'years',st:1}];
function fmtv(f,v){var st=FN[f].st; return st<.1?v.toFixed(2):st<1?v.toFixed(1):String(Math.round(v))}
function pct(v,d){return (v*100).toFixed(d==null?1:d)+'%'}
function mean(a){var s=0;for(var i=0;i<a.length;i++)s+=a[i];return a.length?s/a.length:0}
function medianOf(a){a=a.slice().sort(function(x,y){return x-y});var n=a.length;return n?(n%2?a[(n-1)/2]:(a[n/2-1]+a[n/2])/2):0}

/* ---------- state ---------- */
var MANUAL_HP={h1:16,h2:8,lr:.01,alpha:.001,overlap:.5};
var MOD={manual:null,b:null,e:null}, RES={manual:null,b:null,e:null}, HP={manual:MANUAL_HP,b:null,e:null}, CVF={manual:null,b:null,e:null};
var P=ROWS[Math.floor(N/3)].slice(0,8), PSRC=Math.floor(N/3);
var GF=1; var G={ov:.5};

/* ---------- worker plumbing ---------- */
function workerMain(){
  var E=engineFactory(),rows,tr,te;
  function fin(hp,seed){var ens=E.trainEnsemble(rows,tr,hp,seed,5);return {ens:ens,m:E.evalEns(ens,rows,te)}}
  self.onmessage=function(ev){var m=ev.data;
    if(m.type==='init'){rows=m.rows;tr=m.train;te=m.test;self.postMessage({type:'ready'})}
    else if(m.type==='pso'){
      var res=E.runPSO({variant:m.variant,swarm:m.swarm,iters:m.iters,seed:m.seed,rows:rows,trainIdx:tr,maxEpochs:120,
        onInit:function(i,n){self.postMessage({type:'initp',i:i,n:n})},
        onIter:function(info){self.postMessage({type:'iter',info:info})}});
      self.postMessage({type:'psodone',fit:res.fit,evals:res.evals,ms:res.ms,hp:res.hp});
      var f=fin(res.hp,m.seed); self.postMessage({type:'final',hp:res.hp,ens:f.ens,m:f.m,fit:res.fit});
    } else if(m.type==='final'){var f2=fin(m.hp,m.seed); self.postMessage({type:'final',hp:m.hp,ens:f2.ens,m:f2.m,fit:m.fit})}
  };
}
var WURL=null, WOK=true;
try{WURL=URL.createObjectURL(new Blob([engineFactory.toString()+';('+workerMain.toString()+')();'],{type:'application/javascript'}))}catch(err){WOK=false}
function spawn(onmsg){
  var w=new Worker(WURL); w.onmessage=function(ev){onmsg(ev.data)}; w.onerror=function(ev){onmsg({type:'error',msg:ev.message})};
  w.postMessage({type:'init',rows:ROWS,train:SPLIT.train,test:SPLIT.test}); return w;
}

/* ---------- manual reference model (trains at load) ---------- */
function startManual(){
  if(!WOK)return;
  var w=spawn(function(m){ if(m.type==='final'){MOD.manual=m.ens;RES.manual=m.m;HP.manual=m.hp;w.terminate();
      document.getElementById('hs-live').textContent=pct(m.m.acc); renderRes(); renderPredict()}});
  w.postMessage({type:'final',hp:MANUAL_HP,seed:42,fit:null});
}

/* ---------- data explorer ---------- */
var D={f:1,clean:true};
(function(){
  var g=document.getElementById('dfeat'); FN.forEach(function(f,i){var b=document.createElement('button'); b.textContent=f.s; b.dataset.v=i; b.setAttribute('aria-pressed',i===D.f); g.appendChild(b)});
  seg('dfeat',function(v){D.f=+v; renderData()});
  document.getElementById('dclean').addEventListener('change',function(){D.clean=this.checked; renderData()});
})();
var htip=document.getElementById('htip');
function renderData(){
  var f=D.f,svg=document.getElementById('hist'),v0=[],v1=[],zc=[0,0,0,0,0,0,0,0];
  ROWS.forEach(function(r){ if(r[f]===0)zc[f]++; if(D.clean&&MISS.indexOf(f)>=0&&r[f]===0)return; (r[8]?v1:v0).push(r[f])});
  svg.innerHTML=''; var all=v0.concat(v1),lo=Math.min.apply(null,all),hi=Math.max.apply(null,all);
  var nb=f===0?(hi-lo+1):14, bw=f===0?1:(hi-lo)/nb, c0=[],c1=[],i; for(i=0;i<nb;i++){c0.push(0);c1.push(0)}
  v0.forEach(function(v){c0[Math.min(nb-1,Math.floor((v-lo)/bw))]++}); v1.forEach(function(v){c1[Math.min(nb-1,Math.floor((v-lo)/bw))]++});
  var s0=c0.map(function(c){return c/Math.max(1,v0.length)}),s1=c1.map(function(c){return c/Math.max(1,v1.length)});
  var W=700,H=300,pl=44,pr=10,pt=14,pb=44,iw=W-pl-pr,ih=H-pt-pb,mx=Math.max.apply(null,s0.concat(s1))*1.12, step=mx>.3?.1:mx>.15?.05:.02;
  function Y(v){return pt+ih*(1-v/mx)}
  for(var t=0;t<=mx;t+=step){se('line',{x1:pl,x2:W-pr,y1:Y(t),y2:Y(t),class:'gl'},svg); var tx=se('text',{x:pl-8,y:Y(t)+4,'text-anchor':'end',class:'ax'},svg); tx.textContent=Math.round(t*100)+'%'}
  var slot=iw/nb;
  for(i=0;i<nb;i++){
    var x=pl+slot*i, w=slot*.4;
    se('rect',{x:x+slot*.08,y:Y(s0[i]),width:w,height:Math.max(0,pt+ih-Y(s0[i])),rx:3,fill:C.cob},svg);
    se('rect',{x:x+slot*.08+w+2,y:Y(s1[i]),width:w,height:Math.max(0,pt+ih-Y(s1[i])),rx:3,fill:C.cor},svg);
    var a=lo+bw*i, b=lo+bw*(i+1), lab=f===0?String(Math.round(a)):fmtv(f,a);
    if(nb<=18&&(f===0||i%2===0)){var l=se('text',{x:x+slot/2,y:H-pb+16,'text-anchor':'middle',class:'ax'},svg); l.textContent=lab}
    (function(i,a,b,x){var h=se('rect',{x:x,y:pt,width:slot,height:ih+4,fill:'transparent'},svg);
      h.addEventListener('pointerenter',function(){var r=svg.getBoundingClientRect(),sc=r.width/W; htip.style.left=((x+slot/2)*sc)+'px'; htip.style.top=(Y(Math.max(s0[i],s1[i]))*sc)+'px'; htip.style.opacity=1;
        htip.textContent=(f===0?lab0(a):fmtv(f,a)+'–'+fmtv(f,b))+': '+c0[i]+' healthy, '+c1[i]+' diabetic'});
      h.addEventListener('pointerleave',function(){htip.style.opacity=0})})(i,a,b,x);
  }
  function lab0(a){return String(Math.round(a))}
  var ul=se('text',{x:W-pr,y:H-2,'text-anchor':'end',class:'ax'},svg); ul.textContent=FN[f].n+(FN[f].u?' ('+FN[f].u+')':'')+'  ·  share of each group';
  var m0=mean(v0),m1=mean(v1);
  document.getElementById('dstats').innerHTML='<div><span>Healthy mean</span><b>'+fmtv(f,m0)+'</b></div><div><span>Diabetic mean</span><b>'+fmtv(f,m1)+'</b></div><div><span>Patients</span><b>'+(v0.length+v1.length)+'</b></div><div><span>Diabetic share</span><b>'+pct(v1.length/(v0.length+v1.length),0)+'</b></div>';
  var tot=ROWS.filter(function(r){return r[8]===1}).length;
  var zs=MISS.map(function(k){var c=ROWS.filter(function(r){return r[k]===0}).length; return FN[k].s+' '+c}).join(' · ');
  document.getElementById('dzeros').textContent='Zeros that really mean "not measured": '+zs+'. Overall '+tot+' of '+N+' patients ('+pct(tot/N,0)+') are diabetic, so a model that always says "healthy" scores '+pct(1-tot/N,0)+'.';
}

/* ---------- patient / granulation ---------- */
var gsrc=document.getElementById('gsrc');
function setPatient(row,src){P=row.slice(0,8);PSRC=src;syncSliders();renderGran();renderPredict()}
(function(){
  var g=document.getElementById('gfeat'); FN.forEach(function(f,i){var b=document.createElement('button'); b.textContent=f.s; b.dataset.v=i; b.setAttribute('aria-pressed',i===GF); g.appendChild(b)});
  seg('gfeat',function(v){GF=+v; renderGran()});
  document.getElementById('gpick').addEventListener('click',function(){var i=Math.floor(Math.random()*N); setPatient(ROWS[i],i)});
  document.getElementById('ov').addEventListener('input',function(){G.ov=this.value/100; document.getElementById('ovv').textContent=G.ov.toFixed(2); renderGran()});
  var mf=document.getElementById('mf'),drag=false;
  function set(ev){var r=mf.getBoundingClientRect(),p=ev.touches?ev.touches[0]:ev,xx=(p.clientX-r.left)/r.width*720,t=Math.max(0,Math.min(1,(xx-40)/660)),
      lo=PREPALL.min[GF],hi=PREPALL.max[GF],st=FN[GF].st,v=lo+(hi-lo)*t; v=Math.round(v/st)*st; P[GF]=+v.toFixed(2); PSRC=null; syncSliders(); renderGran(); renderPredict()}
  mf.addEventListener('pointerdown',function(ev){drag=true;mf.setPointerCapture(ev.pointerId);set(ev)});
  mf.addEventListener('pointermove',function(ev){if(drag)set(ev)});
  mf.addEventListener('pointerup',function(){drag=false});
})();
function renderGran(){
  var mf=document.getElementById('mf'); mf.innerHTML=''; var f=GF,W=720,H=280,pl=40,pr=20,pt=16,pb=40,iw=W-pl-pr,ih=H-pt-pb;
  function X(x){return pl+iw*x} function Y(m){return pt+ih*(1-m)}
  var hw=ENG.halfWidth(G.ov), cols=[C.cob,C.ink2,C.cor], names=['low','medium','high'];
  [0,.5,1].forEach(function(g){se('line',{x1:pl,x2:W-pr,y1:Y(g),y2:Y(g),stroke:C.line},mf); var t=se('text',{x:pl-8,y:Y(g)+4,'text-anchor':'end',class:'ax'},mf); t.textContent=g});
  [0,.5,1].forEach(function(c,i){
    var area=[X(Math.max(0,c-hw))+','+Y(0), X(c)+','+Y(1), X(Math.min(1,c+hw))+','+Y(0)];
    // clip triangle to [0,1] by sampling
    var pts=[]; for(var k=0;k<=100;k++){var x=k/100; pts.push(X(x).toFixed(1)+','+Y(ENG.tri(x,c,hw)).toFixed(1))}
    se('polygon',{points:X(0)+','+Y(0)+' '+pts.join(' ')+' '+X(1)+','+Y(0),fill:cols[i],opacity:.12},mf);
    var lo2=Math.max(0,c-hw),hi2=Math.min(1,c+hw),seg2=[X(lo2).toFixed(1)+','+Y(0).toFixed(1)]; for(var q=0;q<=100;q++){var xq=q/100; if(xq>lo2&&xq<hi2)seg2.push(X(xq).toFixed(1)+','+Y(ENG.tri(xq,c,hw)).toFixed(1))} seg2.push(X(hi2).toFixed(1)+','+Y(0).toFixed(1));
    se('polyline',{points:seg2.join(' '),fill:'none',stroke:cols[i],'stroke-width':2.5,'stroke-linejoin':'round'},mf);
    var t=se('text',{x:X(c),y:Y(1)-6,'text-anchor':c===0?'start':c===1?'end':'middle',class:'ax',fill:cols[i]},mf); t.textContent=names[i];
  });
  var lo=PREPALL.min[f],hi=PREPALL.max[f];
  [0,.25,.5,.75,1].forEach(function(g){var t=se('text',{x:X(g),y:H-16,'text-anchor':'middle',class:'ax'},mf); t.textContent=fmtv(f,lo+(hi-lo)*g)});
  var u=se('text',{x:W-pr,y:H-2,'text-anchor':'end',class:'ax'},mf); u.textContent=FN[f].n+(FN[f].u?' ('+FN[f].u+')':'');
  var S=ENG.scaled(PREPALL,P), x0=S[f], m=[0,.5,1].map(function(c){return ENG.tri(x0,c,hw)});
  se('line',{x1:X(x0),x2:X(x0),y1:pt,y2:pt+ih,stroke:C.ink,'stroke-width':1.5,'stroke-dasharray':'4 3'},mf);
  m.forEach(function(v,i){if(v>0)se('circle',{cx:X(x0),cy:Y(v),r:6,fill:cols[i],stroke:C.tile,'stroke-width':2},mf)});
  se('circle',{cx:X(x0),cy:pt+ih,r:9,fill:C.ink,stroke:C.tile,'stroke-width':3,style:'cursor:ew-resize'},mf);
  // table of all 8 features
  var h='<div></div><div class="h">low</div><div class="h">medium</div><div class="h">high</div>';
  for(var k=0;k<8;k++){
    var g=[0,.5,1].map(function(c){return ENG.tri(S[k],c,hw)});
    h+='<div class="nm'+(k===f?' sel':'')+'" data-k="'+k+'">'+FN[k].s+'</div>';
    g.forEach(function(v){h+='<div class="c"><i style="width:'+(v*100)+'%"></i><u>'+v.toFixed(2)+'</u></div>'});
  }
  var ft=document.getElementById('ftbl'); ft.innerHTML=h;
  [].forEach.call(ft.querySelectorAll('.nm'),function(el){el.addEventListener('click',function(){GF=+el.dataset.k;
    [].forEach.call(document.querySelectorAll('#gfeat button'),function(b){b.setAttribute('aria-pressed',+b.dataset.v===GF)}); renderGran()})});
  var raw=P[f]; var zero=MISS.indexOf(f)>=0&&raw===0;
  gsrc.textContent=(PSRC!=null?'Patient record #'+(PSRC+1)+': ':'Edited patient: ')+FN[f].n+' = '+fmtv(f,raw)+(FN[f].u?' '+FN[f].u:'')+(zero?' (stored as 0, treated as missing and replaced by the median)':'')+', scaled to '+x0.toFixed(2)+'.'+(m[0]+m[1]+m[2]===0?' It falls in a gap: no granule fires for this feature.':'');
}

/* ---------- live tuning ---------- */
var T={running:false,w:{},hist:{b:[],e:[]},wh:{b:[],e:[]},st:{b:null,e:null},last:{b:null,e:null},t0:0,done:{b:false,e:false},iters:14,swarm:10};
var VAR={b:{name:'Baseline PSO',sub:'inertia w = 0.72 (fixed)',col:'cob',el:'vb'},e:{name:'Enhanced PSO',sub:'w = 0.4 + 0.5 · diversity',col:'cor',el:'ve'}};
function hpStr(hp){return (hp.h1+(hp.h2>0?' → '+hp.h2:''))+' hidden · lr '+hp.lr.toFixed(4)+' · α '+hp.alpha.toExponential(1)+' · overlap '+hp.overlap.toFixed(2)}
function paintVar(k){
  var v=VAR[k],s=T.st[k],el=document.getElementById(v.el),col=k==='b'?'var(--cobalt)':'var(--coral)';
  var prog=0,txt='Waiting to start.',stat=['–','–','–'],hp='–';
  if(s){
    if(s.phase==='init'){prog=(s.i/s.n)*8; txt='Scoring the initial swarm: particle '+s.i+' of '+s.n}
    else if(s.phase==='iter'){prog=8+92*(s.iter/T.iters); txt='Iteration '+s.iter+' of '+T.iters+'  ·  diversity '+s.div.toFixed(2)}
    else if(s.phase==='train'){prog=100; txt='Search finished. Training a 5-seed ensemble on the best settings…'}
    else if(s.phase==='done'){prog=100; txt='Finished.'}
    else if(s.phase==='stopped'){prog=100; txt='Stopped early.'}
    if(s.best!=null){stat=[pct(s.best),s.w!=null?s.w.toFixed(2):'–',s.evals||'–']}
    if(s.hp)hp=hpStr(s.hp);
  }
  el.innerHTML='<div class="hdr"><div><h3><span class="chip" style="margin-right:8px"><i class="'+v.col+'"></i></span>'+v.name+'</h3><p class="sub">'+v.sub+'</p></div></div>'+
    '<div class="prog"><i style="width:'+prog+'%;background:'+col+'"></i></div><p class="sub">'+txt+'</p>'+
    '<div class="stat"><div><span>Best CV accuracy</span><b>'+stat[0]+'</b></div><div><span>Inertia w</span><b>'+stat[1]+'</b></div><div><span>Networks trained</span><b>'+stat[2]+'</b></div></div>'+
    '<div class="hp" title="Best hyperparameters found so far">'+hp+'</div>';
}
function lineChart(svg,W,H,series,o){
  svg.innerHTML=''; var pl=46,pr=12,pt=10,pb=30,iw=W-pl-pr,ih=H-pt-pb;
  function X(i){return pl+iw*(o.x1>o.x0?(i-o.x0)/(o.x1-o.x0):0)} function Y(v){return pt+ih*(1-(v-o.y0)/(o.y1-o.y0))}
  o.yt.forEach(function(v){se('line',{x1:pl,x2:W-pr,y1:Y(v),y2:Y(v),class:'gl'},svg); var t=se('text',{x:pl-8,y:Y(v)+4,'text-anchor':'end',class:'ax'},svg); t.textContent=o.yf(v)});
  o.xt.forEach(function(i){var t=se('text',{x:X(i),y:H-10,'text-anchor':'middle',class:'ax'},svg); t.textContent=i});
  series.forEach(function(s){ if(!s.d.length)return;
    se('polyline',{points:s.d.map(function(v,i){return X(i+1).toFixed(1)+','+Y(v).toFixed(1)}).join(' '),fill:'none',stroke:s.c,'stroke-width':2.5,'stroke-linejoin':'round','stroke-linecap':'round'},svg);
    se('circle',{cx:X(s.d.length),cy:Y(s.d[s.d.length-1]),r:5,fill:s.c,stroke:C.tile,'stroke-width':2},svg)});
}
function renderLive(){
  var n=T.iters, all=T.hist.b.concat(T.hist.e), lo=.7,hi=.82;
  if(all.length){lo=Math.floor((Math.min.apply(null,all)-.01)*100)/100; hi=Math.ceil((Math.max.apply(null,all)+.01)*100)/100; if(hi-lo<.04){hi=lo+.04}}
  var yt=[],st=hi-lo>.1?.04:.01; for(var v=Math.ceil(lo/st)*st;v<=hi+1e-9;v+=st)yt.push(+v.toFixed(3));
  var xt=[]; for(var i=1;i<=n;i+=Math.max(1,Math.round(n/6)))xt.push(i); if(xt[xt.length-1]!==n)xt.push(n);
  lineChart(document.getElementById('lc'),600,300,[{d:T.hist.b,c:C.cob},{d:T.hist.e,c:C.cor}],{x0:1,x1:Math.max(2,n),y0:lo,y1:hi,yt:yt,yf:function(v){return (v*100).toFixed(1)+'%'},xt:xt});
  lineChart(document.getElementById('lw'),380,300,[{d:T.wh.b,c:C.cob},{d:T.wh.e,c:C.cor}],{x0:1,x1:Math.max(2,n),y0:.3,y1:.95,yt:[.4,.5,.6,.7,.8,.9],yf:function(v){return v.toFixed(1)},xt:xt});
  paintVar('b'); paintVar('e');
}
function tuneEst(){var s=+document.getElementById('sw').value,i=+document.getElementById('it').value; document.getElementById('est').textContent='About '+(s*(i+1))+' networks to cross-validate per swarm, roughly '+Math.max(3,Math.round(s*(i+1)*.1*1.1))+' to '+Math.max(5,Math.round(s*(i+1)*.1*1.8))+' seconds on a laptop (both swarms run in parallel).'}
['sw','it','sd'].forEach(function(id){document.getElementById(id).addEventListener('input',function(){document.getElementById(id+'v').textContent=this.value; tuneEst()})});
tuneEst();
function setBusy(b){T.running=b; document.getElementById('run').disabled=b; document.getElementById('stop').disabled=!b; ['sw','it','sd'].forEach(function(id){document.getElementById(id).disabled=b})}
function onMsg(k,m){
  if(m.type==='error'){T.st[k]={phase:'stopped',best:null};paintVar(k);return}
  if(m.type==='initp'){T.st[k]={phase:'init',i:m.i,n:m.n,best:null}}
  else if(m.type==='iter'){var inf=m.info; T.last[k]=inf; T.hist[k].push(inf.best); T.wh[k].push(inf.w); T.st[k]={phase:'iter',iter:inf.iter,div:inf.div,best:inf.best,w:inf.w,evals:inf.evals,hp:inf.hp}; renderLive()}
  else if(m.type==='psodone'){var s=T.st[k]||{}; T.st[k]={phase:'train',best:m.fit,w:s.w,evals:m.evals,hp:m.hp}}
  else if(m.type==='final'){T.st[k]={phase:T.stopped?'stopped':'done',best:m.fit,w:(T.st[k]||{}).w,evals:(T.st[k]||{}).evals,hp:m.hp};
    MOD[k]=m.ens;RES[k]=m.m;HP[k]=m.hp;CVF[k]=m.fit; T.done[k]=true; if(T.w[k]){T.w[k].terminate();T.w[k]=null}
    renderRes(); renderPredict(); if(T.done.b&&T.done.e)setBusy(false)}
  paintVar(k);
}
function run(){
  if(!WOK||T.running)return;
  T.swarm=+document.getElementById('sw').value; T.iters=+document.getElementById('it').value; var seed=+document.getElementById('sd').value;
  T.hist={b:[],e:[]};T.wh={b:[],e:[]};T.last={b:null,e:null};T.done={b:false,e:false};T.stopped=false;T.st={b:{phase:'init',i:0,n:T.swarm,best:null},e:{phase:'init',i:0,n:T.swarm,best:null}};
  MOD.b=MOD.e=null;RES.b=RES.e=null;HP.b=HP.e=null; setBusy(true); renderLive(); renderRes(); renderPredict();
  ['b','e'].forEach(function(k){var w=spawn(function(m){onMsg(k,m)}); T.w[k]=w; w.postMessage({type:'pso',variant:k==='b'?'baseline':'adaptive',swarm:T.swarm,iters:T.iters,seed:seed})});
}
function stopRun(){
  if(!T.running)return; T.stopped=true; var seed=+document.getElementById('sd').value;
  ['b','e'].forEach(function(k){
    if(T.done[k])return; var w=T.w[k]; if(w){w.terminate();T.w[k]=null}
    var inf=T.last[k]; if(!inf){T.done[k]=true; T.st[k]={phase:'stopped',best:null}; paintVar(k); return}
    T.st[k]={phase:'train',best:inf.best,w:inf.w,evals:inf.evals,hp:inf.hp}; paintVar(k);
    var w2=spawn(function(m){onMsg(k,m)}); T.w[k]=w2; w2.postMessage({type:'final',hp:inf.hp,seed:seed,fit:inf.best});
  });
  if(T.done.b&&T.done.e)setBusy(false);
}
document.getElementById('run').addEventListener('click',run); document.getElementById('stop').addEventListener('click',stopRun);

/* ---------- results tiles ---------- */
var RINFO=[{k:'manual',n:'GNN, manual tuning',c:'var(--ink-3)'},{k:'b',n:'GNN + baseline PSO',c:'var(--cobalt)'},{k:'e',n:'GNN + enhanced PSO',c:'var(--coral)'}];
function renderRes(){
  var h='';
  RINFO.forEach(function(r){
    var m=RES[r.k],hp=HP[r.k];
    if(!m){var busy=(r.k==='manual')?'Training the reference model…':(T.running?'Search in progress…':'Run both swarms to fill this in.');
      h+='<div class="rt pend"><h3><span class="chip" style="margin-right:8px"><i style="background:'+r.c+'"></i></span>'+r.n+'</h3><div class="big2" style="color:var(--ink-3)">--</div><p class="sub">'+busy+'</p></div>'; return}
    h+='<div class="rt"><h3><span class="chip" style="margin-right:8px"><i style="background:'+r.c+'"></i></span>'+r.n+'</h3>'+
      '<div class="big2">'+pct(m.acc)+'</div><p class="sub">test accuracy on 154 unseen patients</p>'+
      '<div class="mm"><div><span>F1</span><b>'+m.f1.toFixed(3)+'</b></div><div><span>AUC</span><b>'+m.auc.toFixed(3)+'</b></div><div><span>Recall</span><b>'+pct(m.recall,0)+'</b></div><div><span>Spec.</span><b>'+pct(m.spec,0)+'</b></div></div>'+
      '<div class="cm2"><div class="g">'+m.tn+'<span>true negative</span></div><div class="r">'+m.fp+'<span>false positive</span></div><div class="r">'+m.fn+'<span>false negative</span></div><div class="g">'+m.tp+'<span>true positive</span></div></div>'+
      '<div class="hp" style="margin-top:12px">'+hpStr(hp)+'</div>'+(CVF[r.k]!=null?'<p class="delta">search CV accuracy '+pct(CVF[r.k])+'</p>':'')+'</div>';
  });
  document.getElementById('res').innerHTML=h;
}

/* ---------- predict ---------- */
(function(){
  var g=document.getElementById('pgrid');
  FN.forEach(function(f,i){
    var d=document.createElement('div'); d.className='ctl'; var lo=PREPALL.min[i],hi=PREPALL.max[i];
    d.innerHTML='<label for="p'+i+'">'+f.n+(f.u?' ('+f.u+')':'')+' <b id="pv'+i+'"></b></label><input id="p'+i+'" type="range" min="'+lo+'" max="'+hi+'" step="'+f.st+'">';
    g.appendChild(d); d.querySelector('input').addEventListener('input',function(){P[i]=+this.value; PSRC=null; document.getElementById('pv'+i).textContent=fmtv(i,P[i]); renderGran(); renderPredict()});
  });
  seg('presets',function(v){
    if(v==='real'){var i=Math.floor(Math.random()*N); setPatient(ROWS[i],i)}
    else{var cls=v==='diab'?1:0,row=[]; for(var f=0;f<8;f++){var vals=ROWS.filter(function(r){return r[8]===cls&&!(MISS.indexOf(f)>=0&&r[f]===0)}).map(function(r){return r[f]}); row.push(medianOf(vals))} setPatient(row,null)}
  });
  [].forEach.call(document.querySelectorAll('#presets button'),function(b){b.addEventListener('click',function(){setTimeout(function(){b.setAttribute('aria-pressed','false')},250)})});
})();
function syncSliders(){for(var i=0;i<8;i++){var el=document.getElementById('p'+i); var v=P[i]; if(MISS.indexOf(i)>=0&&v===0)v=PREPALL.med[i]; el.value=v; document.getElementById('pv'+i).textContent=fmtv(i,+v)}}
function renderPredict(){
  var h='',first=null;
  RINFO.forEach(function(r){
    var ens=MOD[r.k];
    if(!ens){h+='<div class="pm"><span>'+r.n.replace('GNN + ','').replace('GNN, ','')+'</span><div class="bar"></div><span class="v" style="color:var(--ink-3)">–</span></div>'; return}
    var p=ENG.predictEns(ens,P).p; if(first==null)first=p;
    h+='<div class="pm"><span>'+r.n.replace('GNN + ','').replace('GNN, ','')+'</span><div class="bar"><i style="width:'+(p*100)+'%;background:'+r.c+'"></i><s></s></div><span class="v">'+pct(p,0)+'</span></div>';
  });
  document.getElementById('pm').innerHTML=h;
  var big=document.getElementById('bigp'),vd=document.getElementById('pverd');
  if(first==null){big.textContent='--'; vd.textContent='Training the reference model…'; return}
  var ps=[]; RINFO.forEach(function(r){if(MOD[r.k])ps.push(ENG.predictEns(MOD[r.k],P).p)});
  var avg=mean(ps); big.textContent=Math.round(avg*100)+'%';
  var actual=PSRC!=null?(ROWS[PSRC][8]?'diabetic':'not diabetic'):null;
  vd.textContent=(avg>=.5?'Leans diabetic':'Leans non-diabetic')+' (average of '+ps.length+' model'+(ps.length>1?'s':'')+', 50% line dashed)'+(actual?'. This is record #'+(PSRC+1)+', whose actual outcome is '+actual+'.':'.');
}

/* ---------- boot ---------- */
function renderAll(){renderData();renderGran();renderLive();renderRes();renderPredict();renderConv();renderCeil();paintRows()}
buildRows(); syncSliders(); renderAll();
if(!WOK){document.getElementById('est').textContent='Web Workers are unavailable in this browser, so live tuning is disabled.'; document.getElementById('run').disabled=true}
else startManual();
window.addEventListener('resize',function(){paintRows()});

// Granular neural network + PSO engine (pure JS, no dependencies).
// Written as a factory so the same source runs on the page and inside a Web Worker.
function engineFactory(){
  "use strict";
  var MISSING=[1,2,3,4,5];           // glucose, bp, skin, insulin, bmi: a zero means "not measured"
  var NF=8, NG=3, D=NF+NF*NG;         // 8 crisp + 24 granule inputs = 32

  function rng(a){return function(){a|=0;a=a+0x6D2B79F5|0;var t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
  function shuffle(arr,r){for(var i=arr.length-1;i>0;i--){var j=Math.floor(r()*(i+1)),t=arr[i];arr[i]=arr[j];arr[j]=t}return arr}

  // rows: array of [8 features..., outcome]
  function stratSplit(rows,frac,seed){
    var r=rng(seed),pos=[],neg=[],te=[],tr=[];
    rows.forEach(function(row,i){(row[8]===1?pos:neg).push(i)});
    [pos,neg].forEach(function(g){shuffle(g,r); var k=Math.round(g.length*frac); g.forEach(function(i,n){(n<k?te:tr).push(i)})});
    return {train:shuffle(tr,r),test:te};
  }
  function median(a){a=a.slice().sort(function(x,y){return x-y}); var n=a.length; return n? (n%2?a[(n-1)/2]:(a[n/2-1]+a[n/2])/2):0}

  function fitPrep(rows,idx){
    var med=[],mn=[],mx=[],f,i;
    for(f=0;f<NF;f++){
      var vals=idx.map(function(k){return rows[k][f]});
      if(MISSING.indexOf(f)>=0){var nz=vals.filter(function(v){return v>0}); med[f]=median(nz); vals=vals.map(function(v){return v>0?v:med[f]})}
      else med[f]=0;
      mn[f]=Math.min.apply(null,vals); mx[f]=Math.max.apply(null,vals);
    }
    return {med:med,min:mn,max:mx};
  }
  function tri(x,c,hw){var v=1-Math.abs(x-c)/hw; return v>0?v:0}
  function halfWidth(overlap){return Math.max(0.04,overlap*0.5)}
  function scaled(prep,row){
    var s=new Float64Array(NF);
    for(var f=0;f<NF;f++){var v=row[f]; if(MISSING.indexOf(f)>=0&&!(v>0)) v=prep.med[f];
      var span=prep.max[f]-prep.min[f]; var t=span>0?(v-prep.min[f])/span:0; s[f]=t<0?0:t>1?1:t}
    return s;
  }
  function granulate(s,overlap){
    var out=new Float64Array(D),hw=halfWidth(overlap),f;
    for(f=0;f<NF;f++){out[f]=s[f]; out[NF+f*3]=tri(s[f],0,hw); out[NF+f*3+1]=tri(s[f],.5,hw); out[NF+f*3+2]=tri(s[f],1,hw)}
    return out;
  }
  function features(prep,row,overlap){return granulate(scaled(prep,row),overlap)}

  function sigmoid(z){return z>=0?1/(1+Math.exp(-z)):Math.exp(z)/(1+Math.exp(z))}
  function initNet(sizes,r){
    var W=[],b=[];
    for(var l=0;l<sizes.length-1;l++){var fi=sizes[l],fo=sizes[l+1],lim=Math.sqrt(6/(fi+fo)),w=new Float64Array(fi*fo);
      for(var k=0;k<w.length;k++)w[k]=(r()*2-1)*lim; W.push(w); b.push(new Float64Array(fo))}
    return {sizes:sizes,W:W,b:b};
  }
  function forward(net,x,acts){
    acts[0]=x; var L=net.sizes.length-1;
    for(var l=0;l<L;l++){
      var fi=net.sizes[l],fo=net.sizes[l+1],W=net.W[l],b=net.b[l],a=acts[l],o=acts[l+1];
      for(var j=0;j<fo;j++)o[j]=b[j];
      for(var i=0;i<fi;i++){var ai=a[i]; if(ai!==0){var base=i*fo; for(var j2=0;j2<fo;j2++)o[j2]+=ai*W[base+j2]}}
      if(l<L-1){for(var j3=0;j3<fo;j3++)if(o[j3]<0)o[j3]=0}
    }
    return sigmoid(acts[L][0]);
  }
  function mkActs(sizes){return sizes.map(function(n){return new Float64Array(n)})}
  function proba(net,X){var acts=mkActs(net.sizes),out=new Float64Array(X.length); for(var i=0;i<X.length;i++){acts[0]=X[i]; out[i]=forward(net,X[i],acts)} return out}
  function cloneNet(n){return {sizes:n.sizes,W:n.W.map(function(w){return Float64Array.from(w)}),b:n.b.map(function(w){return Float64Array.from(w)})}}

  // Adam + L2 + early stopping on a 10% validation slice (mirrors sklearn's MLPClassifier defaults)
  function train(X,y,hp,seed,maxEpochs){
    var r=rng(seed),n=X.length,sizes=[D,hp.h1]; if(hp.h2>0)sizes.push(hp.h2); sizes.push(1);
    var net=initNet(sizes,r),L=sizes.length-1;
    var order=shuffle(Array.apply(null,Array(n)).map(function(_,i){return i}),r), nv=Math.max(8,Math.round(n*0.1));
    var vi=order.slice(0,nv), ti=order.slice(nv), nt=ti.length;
    var mW=net.W.map(function(w){return new Float64Array(w.length)}),vW=net.W.map(function(w){return new Float64Array(w.length)});
    var mB=net.b.map(function(w){return new Float64Array(w.length)}),vB=net.b.map(function(w){return new Float64Array(w.length)});
    var gW=net.W.map(function(w){return new Float64Array(w.length)}),gB=net.b.map(function(w){return new Float64Array(w.length)});
    var acts=mkActs(sizes),deltas=sizes.map(function(s){return new Float64Array(s)});
    var B=Math.min(200,nt),t=0,best=-1,bestNet=null,stall=0,b1=.9,b2=.999,eps=1e-8,lr=hp.lr,alpha=hp.alpha;
    for(var ep=0;ep<maxEpochs;ep++){
      shuffle(ti,r);
      for(var s0=0;s0<nt;s0+=B){
        var e0=Math.min(nt,s0+B),bs=e0-s0,l,k;
        for(l=0;l<L;l++){gW[l].fill(0);gB[l].fill(0)}
        for(var q=s0;q<e0;q++){
          var id=ti[q],p=forward(net,X[id],acts);
          deltas[L][0]=p-y[id];
          for(l=L-1;l>=0;l--){
            var fi=sizes[l],fo=sizes[l+1],W=net.W[l],a=acts[l],d=deltas[l+1],gw=gW[l],gb=gB[l];
            for(var j=0;j<fo;j++)gb[j]+=d[j];
            for(var i=0;i<fi;i++){var ai=a[i],base=i*fo,acc=0;
              for(var j2=0;j2<fo;j2++){gw[base+j2]+=ai*d[j2]; acc+=W[base+j2]*d[j2]}
              if(l>0)deltas[l][i]=ai>0?acc:0;}
          }
        }
        t++; var c1=1-Math.pow(b1,t),c2=1-Math.pow(b2,t),step=lr*Math.sqrt(c2)/c1;
        for(l=0;l<L;l++){
          var w=net.W[l],g=gW[l],m=mW[l],v=vW[l];
          for(k=0;k<w.length;k++){var gr=(g[k]+alpha*w[k])/bs; m[k]=b1*m[k]+(1-b1)*gr; v[k]=b2*v[k]+(1-b2)*gr*gr; w[k]-=step*m[k]/(Math.sqrt(v[k])+eps)}
          var bb=net.b[l],gbb=gB[l],mb=mB[l],vb=vB[l];
          for(k=0;k<bb.length;k++){var gr2=gbb[k]/bs; mb[k]=b1*mb[k]+(1-b1)*gr2; vb[k]=b2*vb[k]+(1-b2)*gr2*gr2; bb[k]-=step*mb[k]/(Math.sqrt(vb[k])+eps)}
        }
      }
      var ok=0; for(var z=0;z<nv;z++){var pp=forward(net,X[vi[z]],acts); if((pp>=.5?1:0)===y[vi[z]])ok++}
      var score=ok/nv;
      if(score>best+1e-4){best=score;bestNet=cloneNet(net);stall=0}else if(++stall>=10)break;
    }
    return bestNet||net;
  }

  function buildX(rows,idx,prep,overlap){return idx.map(function(k){return features(prep,rows[k],overlap)})}
  function labels(rows,idx){return Float64Array.from(idx.map(function(k){return rows[k][8]}))}

  function cvFitness(rows,trainIdx,hp,seed,maxEpochs){
    var r=rng(seed),pos=[],neg=[]; trainIdx.forEach(function(i){(rows[i][8]===1?pos:neg).push(i)});
    shuffle(pos,r);shuffle(neg,r); var K=3,folds=[[],[],[]]; pos.forEach(function(i,n){folds[n%K].push(i)}); neg.forEach(function(i,n){folds[n%K].push(i)});
    var tot=0;
    for(var f=0;f<K;f++){
      var va=folds[f],tr=[]; for(var g=0;g<K;g++)if(g!==f)tr=tr.concat(folds[g]);
      var prep=fitPrep(rows,tr),Xtr=buildX(rows,tr,prep,hp.overlap),net=train(Xtr,labels(rows,tr),hp,seed+f,maxEpochs);
      var Xva=buildX(rows,va,prep,hp.overlap),pr=proba(net,Xva),ok=0; for(var i=0;i<va.length;i++)if((pr[i]>=.5?1:0)===rows[va[i]][8])ok++;
      tot+=ok/va.length;
    }
    return tot/K;
  }

  function auc(y,p){
    var n=y.length,idx=Array.apply(null,Array(n)).map(function(_,i){return i}).sort(function(a,b){return p[a]-p[b]}),rk=new Float64Array(n);
    for(var i=0;i<n;){var j=i;while(j+1<n&&p[idx[j+1]]===p[idx[i]])j++; var r=(i+j)/2+1; for(var k=i;k<=j;k++)rk[idx[k]]=r; i=j+1}
    var np=0,sr=0; for(var q=0;q<n;q++)if(y[q]===1){np++;sr+=rk[q]}
    var nn=n-np; return np&&nn?(sr-np*(np+1)/2)/(np*nn):0.5;
  }
  function metrics(y,p){
    var tp=0,fp=0,tn=0,fn=0; for(var i=0;i<y.length;i++){var c=p[i]>=.5?1:0; if(c&&y[i])tp++; else if(c)fp++; else if(y[i])fn++; else tn++}
    var prec=tp+fp?tp/(tp+fp):0,rec=tp+fn?tp/(tp+fn):0;
    return {acc:(tp+tn)/y.length,f1:prec+rec?2*prec*rec/(prec+rec):0,auc:auc(y,p),tp:tp,fp:fp,tn:tn,fn:fn,recall:rec,spec:tn+fp?tn/(tn+fp):0};
  }

  // A trained model = prep (imputation + scaling) + overlap + net. Self-contained, structured-clone friendly.
  function trainModel(rows,trainIdx,hp,seed,maxEpochs){
    var prep=fitPrep(rows,trainIdx),X=buildX(rows,trainIdx,prep,hp.overlap),net=train(X,labels(rows,trainIdx),hp,seed,maxEpochs||200);
    return {prep:prep,hp:hp,net:net};
  }
  function evaluate(model,rows,idx){
    var X=buildX(rows,idx,model.prep,model.hp.overlap),p=proba(model.net,X); return metrics(labels(rows,idx),p);
  }
  function predictRow(model,row){
    var x=features(model.prep,row,model.hp.overlap),p=forward(model.net,x,mkActs(model.net.sizes));
    return {p:p,granules:Array.prototype.slice.call(x,NF),crisp:Array.prototype.slice.call(x,0,NF)};
  }


  // A 5-seed ensemble keeps the held-out numbers from swinging on a single lucky/unlucky initialisation.
  function trainEnsemble(rows,idx,hp,seed,k){var ms=[];for(var i=0;i<k;i++)ms.push(trainModel(rows,idx,hp,seed+i*101,200));return {models:ms,hp:hp}}
  function ensProba(ens,rows,idx){
    var out=new Float64Array(idx.length);
    ens.models.forEach(function(m){var p=proba(m.net,buildX(rows,idx,m.prep,m.hp.overlap));for(var i=0;i<p.length;i++)out[i]+=p[i]/ens.models.length});
    return out;
  }
  function evalEns(ens,rows,idx){return metrics(labels(rows,idx),ensProba(ens,rows,idx))}
  function predictEns(ens,row){
    var p=0,first=null; ens.models.forEach(function(m){var r=predictRow(m,row); p+=r.p/ens.models.length; if(!first)first=r});
    return {p:p,granules:first.granules,crisp:first.crisp,overlap:ens.hp.overlap};
  }

  // ---------- PSO ----------
  var BOUNDS=[[8,48],[0,24],[-3,-1],[-5,-1],[0.15,1]];
  function decode(x){return {h1:Math.round(x[0]),h2:x[1]<4?0:Math.round(x[1]),lr:Math.pow(10,x[2]),alpha:Math.pow(10,x[3]),overlap:x[4]}}
  function diversity(P){
    var d=BOUNDS.length,c=new Float64Array(d),n=P.length,i,k;
    P.forEach(function(p){for(k=0;k<d;k++)c[k]+=(p.x[k]-BOUNDS[k][0])/(BOUNDS[k][1]-BOUNDS[k][0])});
    for(k=0;k<d;k++)c[k]/=n; var s=0;
    P.forEach(function(p){var q=0;for(k=0;k<d;k++){var u=(p.x[k]-BOUNDS[k][0])/(BOUNDS[k][1]-BOUNDS[k][0])-c[k];q+=u*u}s+=Math.sqrt(q)}); return s/n;
  }
  // opts: {variant:'baseline'|'adaptive', swarm, iters, seed, rows, trainIdx, maxEpochs, onIter(info), shouldStop()}
  function runPSO(o){
    var r=rng(o.seed),d=BOUNDS.length,P=[],gBest=null,gFit=-1,evals=0,C=1.49,adaptive=o.variant==='adaptive',hist=[],t0=Date.now();
    function ev(x){evals++; return cvFitness(o.rows,o.trainIdx,decode(x),o.seed+evals,o.maxEpochs||120)}
    for(var i=0;i<o.swarm;i++){
      var x=BOUNDS.map(function(b){return b[0]+r()*(b[1]-b[0])}),v=BOUNDS.map(function(b){return (r()*2-1)*(b[1]-b[0])*.1});
      var f=ev(x); P.push({x:x,v:v,bx:x.slice(),bf:f}); if(f>gFit){gFit=f;gBest=x.slice()}
      if(o.onInit)o.onInit(i+1,o.swarm);
    }
    var d0=Math.max(1e-9,diversity(P));
    for(var it=1;it<=o.iters;it++){
      if(o.shouldStop&&o.shouldStop())break;
      var div=diversity(P)/d0,dv=Math.min(1,div),w=adaptive?0.4+0.5*dv:0.72;
      for(var pi=0;pi<P.length;pi++){
        var p=P[pi];
        for(var k=0;k<d;k++){
          var vmax=(BOUNDS[k][1]-BOUNDS[k][0])*.25,r1=r(),r2=r();
          p.v[k]=w*p.v[k]+C*r1*(p.bx[k]-p.x[k])+C*r2*(gBest[k]-p.x[k]);
          p.v[k]=Math.max(-vmax,Math.min(vmax,p.v[k]));
          p.x[k]=Math.max(BOUNDS[k][0],Math.min(BOUNDS[k][1],p.x[k]+p.v[k]));
        }
        var fit=ev(p.x); if(fit>p.bf){p.bf=fit;p.bx=p.x.slice()} if(fit>gFit){gFit=fit;gBest=p.x.slice()}
      }
      hist.push(gFit);
      if(o.onIter)o.onIter({iter:it,best:gFit,w:w,div:dv,evals:evals,hp:decode(gBest),ms:Date.now()-t0,fits:P.map(function(p){return p.bf})});
    }
    return {bestX:gBest,hp:decode(gBest),fit:gFit,hist:hist,evals:evals,ms:Date.now()-t0};
  }

  return {rng:rng,stratSplit:stratSplit,trainEnsemble:trainEnsemble,evalEns:evalEns,predictEns:predictEns,trainModel:trainModel,evaluate:evaluate,predictRow:predictRow,runPSO:runPSO,decode:decode,BOUNDS:BOUNDS,
          halfWidth:halfWidth,fitPrep:fitPrep,scaled:scaled,tri:tri,cvFitness:cvFitness,NF:NF};
}
if(typeof module!=='undefined')module.exports=engineFactory;

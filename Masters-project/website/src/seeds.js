const fs=require('fs');const E=require('./engine.js')();
const rows=fs.readFileSync('../dl/Datasets/pima-indians-diabetes.data.csv','utf8').trim().split('\n').map(l=>l.split(',').map(Number));
const sp=E.stratSplit(rows,0.2,42);const seed=+process.argv[2];
const out={seed};
for(const v of ['baseline','adaptive']){
  const r=E.runPSO({variant:v,swarm:10,iters:14,seed,rows,trainIdx:sp.train,maxEpochs:120});
  const ens=E.trainEnsemble(rows,sp.train,r.hp,seed,5),m=E.evalEns(ens,rows,sp.test);
  const first=r.hist.findIndex(x=>x>=r.fit-1e-9)+1;
  out[v]={cv:+r.fit.toFixed(3),conv:first,acc:+m.acc.toFixed(3),f1:+m.f1.toFixed(3),auc:+m.auc.toFixed(3),hp:r.hp};
}
console.log(JSON.stringify(out));

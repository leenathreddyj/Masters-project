"""Run baseline vs adaptive PSO, train a 5-seed ensemble on the best settings, evaluate on the held-out test set.
Usage:  python src/run_experiments.py --swarm 25 --iters 80 --seeds 0 1 2   (full study)
        python src/run_experiments.py --swarm 8 --iters 5 --seeds 0        (quick smoke test)"""
import argparse, json, os, sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
sys.path.insert(0, os.path.dirname(__file__))
from granular import load_data, split, GNN
from pso import run_pso

MANUAL = dict(h1=16, h2=8, lr=0.01, alpha=1e-3, overlap=0.5)


def ensemble_eval(hp, Xtr, ytr, Xte, yte, seed, k=5):
    p = np.mean([GNN(**hp, seed=seed + 101 * i).fit(Xtr, ytr).predict_proba(Xte) for i in range(k)], axis=0)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    return dict(acc=accuracy_score(yte, pred), f1=f1_score(yte, pred), auc=roc_auc_score(yte, p), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--swarm", type=int, default=25); ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--out", default="results/results.json")
    a = ap.parse_args()
    X, y = load_data(); Xtr, Xte, ytr, yte = split(X, y, seed=42)
    out = {"split": {"train": len(ytr), "test": len(yte)}, "manual": ensemble_eval(MANUAL, Xtr, ytr, Xte, yte, 42), "runs": []}
    print("manual GNN:", {k: round(v, 3) for k, v in out["manual"].items()})
    for s in a.seeds:
        for v in ("baseline", "adaptive"):
            r = run_pso(Xtr, ytr, v, a.swarm, a.iters, seed=s, verbose=False)
            m = ensemble_eval(r["hp"], Xtr, ytr, Xte, yte, s)
            out["runs"].append(dict(seed=s, variant=v, fitness=r["fitness"], history=r["history"], weights=r["weights"], hp=r["hp"], test=m))
            print(f"seed {s} {v:8s} CV {r['fitness']:.3f} | test acc {m['acc']:.3f} f1 {m['f1']:.3f} auc {m['auc']:.3f}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)

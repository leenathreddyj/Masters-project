"""Particle swarm optimisation over the 5 GNN hyperparameters.
baseline: constant inertia w = 0.72.  adaptive: w = w_min + (w_max - w_min) * min(1, diversity / initial diversity)."""
import numpy as np
from granular import cv_accuracy

BOUNDS = np.array([[8, 48],      # hidden layer 1 width
                   [0, 24],      # hidden layer 2 width (<4 means single hidden layer)
                   [-3, -1],     # log10 learning rate
                   [-5, -1],     # log10 L2 alpha
                   [0.15, 1.0]]) # granule overlap


def decode(x):
    return dict(h1=int(round(x[0])), h2=0 if x[1] < 4 else int(round(x[1])), lr=10 ** x[2], alpha=10 ** x[3], overlap=float(x[4]))


def diversity(P):
    u = (P - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
    return float(np.mean(np.linalg.norm(u - u.mean(0), axis=1)))


def run_pso(X, y, variant="baseline", swarm=25, iters=80, seed=0, c1=1.49, c2=1.49, w_fixed=0.72, w_min=0.4, w_max=0.9, verbose=False):
    rng = np.random.default_rng(seed)
    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    P = BOUNDS[:, 0] + rng.random((swarm, 5)) * span
    V = (rng.random((swarm, 5)) * 2 - 1) * span * 0.1
    evals = 0

    def fit(x):
        nonlocal evals
        evals += 1
        return cv_accuracy(X, y, decode(x), seed=seed + evals)

    pbest, pfit = P.copy(), np.array([fit(p) for p in P])
    g = int(pfit.argmax()); gbest, gfit = pbest[g].copy(), float(pfit[g])
    d0 = max(diversity(P), 1e-9)
    history, weights = [], []
    for it in range(1, iters + 1):
        dv = min(1.0, diversity(P) / d0)
        w = w_fixed if variant == "baseline" else w_min + (w_max - w_min) * dv
        for i in range(swarm):
            r1, r2 = rng.random(5), rng.random(5)
            V[i] = np.clip(w * V[i] + c1 * r1 * (pbest[i] - P[i]) + c2 * r2 * (gbest - P[i]), -0.25 * span, 0.25 * span)
            P[i] = np.clip(P[i] + V[i], BOUNDS[:, 0], BOUNDS[:, 1])
            f = fit(P[i])
            if f > pfit[i]:
                pfit[i], pbest[i] = f, P[i].copy()
            if f > gfit:
                gfit, gbest = f, P[i].copy()
        history.append(gfit); weights.append(w)
        if verbose:
            print(f"[{variant}] iter {it:3d} best CV acc {gfit:.4f} w={w:.3f}")
    return dict(hp=decode(gbest), fitness=gfit, history=history, weights=weights, evals=evals)

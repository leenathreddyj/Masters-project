# Enhanced PSO for Hyperparameter Tuning in Granular Neural Networks (Diabetes Prediction)

Master's final project, Venkata Saileenath Reddy Jampala.

A granular neural network (GNN) predicts diabetes from the Pima Indians dataset (768 patients, 8 features). Each scaled feature is
expressed as three triangular fuzzy granules (low / medium / high), giving 8 crisp + 24 granule = 32 inputs. Five hyperparameters
(hidden widths, learning rate, L2 alpha, granule overlap) are tuned by particle swarm optimisation, comparing a fixed-inertia
**baseline** (w = 0.72) with a **diversity-adaptive** inertia variant (w = 0.4 + 0.5 * diversity).

## Repository layout
| Path | Contents |
|---|---|
| `website/index.html` | Interactive site: the real GNN and both PSO variants run live in the browser (Web Workers). Open the file in a browser. |
| `website/src/` | Source of the site: JavaScript engine (`engine.js`), UI (`app.js`), build script. `node seeds.js 11` runs a headless tuning job. |
| `src/` | Python pipeline: `granular.py` (preprocessing, granulation, GNN), `pso.py`, `run_experiments.py`. |
| `data/` | `pima-indians-diabetes.data.csv` |
| `docs/` | Project report (`.docx`) and defense slides (`.pptx`) |

## Run the Python pipeline
```
pip install -r requirements.txt
python src/run_experiments.py --swarm 8 --iters 5 --seeds 0          # quick check, about 10 seconds
python src/run_experiments.py --swarm 25 --iters 80 --seeds 0 1 2    # full-size study (slow)
```
Results are written to `results/results.json`.

## Method
- Preprocessing: zeros in glucose, blood pressure, skin thickness, insulin and BMI are treated as missing, filled with the training median; min-max scaling. Fit on training data only (and refit inside every CV fold).
- Split: stratified 80/20 (seed 42): 614 train / 154 test.
- PSO fitness: 3-fold stratified CV accuracy on the training split. c1 = c2 = 1.49, velocity clamp 25% of each range.
- Reporting: 5-seed ensemble (averaged probabilities) scored once on the held-out test set.

## Key findings (from the report)
- The adaptive-inertia variant did not beat the fixed-inertia baseline (CV fitness 0.799 vs 0.802; convergence iteration 11.7 vs 8.7).
- PSO-tuned models were roughly equal to a hand-tuned GNN (about 73% vs 74% test accuracy).
- Every model family tried plateaus at 71-76% test accuracy on this dataset; the proposal's 85-90% target is not reachable here.

## Full-size run of `src/` (swarm 25, 80 iterations, seeds 0-2)
Raw output: `results/results.json`. Means over 3 seeds, scored on the 154 held-out patients (5-seed ensembles):

| Model | CV fitness | Test accuracy | F1 | AUC |
|---|---|---|---|---|
| Manual GNN (16, 8, lr 0.01, alpha 1e-3, overlap 0.5) | n/a | 75.3% | 0.612 | 0.821 |
| GNN + baseline PSO (w = 0.72) | 0.799 | 72.9% | 0.602 | 0.808 |
| GNN + adaptive PSO | 0.799 | 73.6% | 0.626 | 0.813 |

Same conclusion as the report: the two PSO variants are statistically indistinguishable, and neither beats a sensible hand-tuned GNN. The adaptive
variant was marginally ahead on test accuracy here (73.6% vs 72.9%) but equal on CV fitness, well within seed noise. These numbers are from this
re-implementation and differ slightly from the report's original runs.

## Note on reproducibility
The report's figures come from the original full-size runs. The code in `src/` and the in-browser engine are clean re-implementations of the same
method, so reruns give similar but not identical numbers, and results vary with the seed.

Research prototype only. Not a diagnostic tool.

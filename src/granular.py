"""Data preparation, fuzzy granulation and the granular neural network (GNN)."""
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

COLS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "pedigree", "age", "outcome"]
ZERO_AS_MISSING = [1, 2, 3, 4, 5]  # glucose, BP, skin, insulin, BMI: a 0 means "not measured"


def load_data(path="data/pima-indians-diabetes.data.csv"):
    df = pd.read_csv(path, header=None, names=COLS)
    return df.iloc[:, :8].to_numpy(float), df["outcome"].to_numpy(int)


def split(X, y, seed=42, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


class Preprocessor:
    """Median imputation of impossible zeros + min-max scaling, fit on training data only."""

    def fit(self, X):
        X = X.copy()
        self.med_ = np.zeros(X.shape[1])
        for j in ZERO_AS_MISSING:
            nz = X[:, j][X[:, j] > 0]
            self.med_[j] = np.median(nz)
            X[X[:, j] == 0, j] = self.med_[j]
        self.min_, self.max_ = X.min(0), X.max(0)
        return self

    def transform(self, X):
        X = X.copy()
        for j in ZERO_AS_MISSING:
            X[X[:, j] <= 0, j] = self.med_[j]
        span = np.where(self.max_ > self.min_, self.max_ - self.min_, 1.0)
        return np.clip((X - self.min_) / span, 0, 1)


def granulate(S, overlap):
    """Each scaled feature -> (low, medium, high) triangular memberships centred at 0, .5, 1.
    Half-width = max(0.04, overlap/2); output = 8 crisp + 24 granule columns = 32 inputs."""
    hw = max(0.04, overlap * 0.5)
    parts = [S]
    for j in range(S.shape[1]):
        parts.append(np.stack([np.clip(1 - np.abs(S[:, j] - c) / hw, 0, 1) for c in (0.0, 0.5, 1.0)], axis=1))
    return np.hstack(parts)


class GNN:
    """Preprocess -> granulate -> MLP (ReLU hidden, sigmoid output, Adam, L2, early stopping)."""

    def __init__(self, h1=16, h2=8, lr=0.01, alpha=1e-3, overlap=0.5, seed=0, max_iter=200):
        self.h1, self.h2, self.lr, self.alpha, self.overlap, self.seed, self.max_iter = h1, h2, lr, alpha, overlap, seed, max_iter

    def fit(self, X, y):
        self.prep_ = Preprocessor().fit(X)
        Z = granulate(self.prep_.transform(X), self.overlap)
        layers = (self.h1,) if self.h2 < 4 else (self.h1, self.h2)
        self.net_ = MLPClassifier(hidden_layer_sizes=layers, activation="relu", solver="adam",
                                  learning_rate_init=self.lr, alpha=self.alpha, batch_size=min(200, len(X)),
                                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4,
                                  max_iter=self.max_iter, random_state=self.seed)
        self.net_.fit(Z, y)
        return self

    def predict_proba(self, X):
        return self.net_.predict_proba(granulate(self.prep_.transform(X), self.overlap))[:, 1]


def cv_accuracy(X, y, hp, seed=0, folds=3, max_iter=120):
    """PSO fitness: stratified k-fold CV accuracy (preprocessing refit inside every fold)."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    acc = []
    for k, (tr, va) in enumerate(skf.split(X, y)):
        m = GNN(**hp, seed=seed + k, max_iter=max_iter).fit(X[tr], y[tr])
        acc.append(np.mean((m.predict_proba(X[va]) >= 0.5) == y[va]))
    return float(np.mean(acc))

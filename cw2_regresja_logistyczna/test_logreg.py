# -*- coding: utf-8 -*-
"""
Testy poprawności implementacji z logreg.py.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu.]

Uruchomienie:  python test_logreg.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from logreg import LogisticRegressionGD, bce, load_data, metrics, sigmoid  # noqa: E402


def test_sigmoid():
    z = np.array([-1000.0, -5.0, 0.0, 5.0, 1000.0])
    s = sigmoid(z)
    assert np.all(np.isfinite(s)), "sigmoid daje NaN/inf dla dużych |z|"
    assert abs(s[2] - 0.5) < 1e-12
    assert abs(s[1] + s[3] - 1.0) < 1e-12, "sigma(-z) = 1 - sigma(z)"
    print("OK  sigmoid: stabilna, sigma(0)=0.5, symetryczna")


def test_gradient():
    """Gradient analityczny  -(1/N) X^T (y - p)  porównany z różnicami skończonymi."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)); y = (rng.random(50) > 0.5).astype(float)
    w = rng.normal(size=4); b = 0.3
    p = sigmoid(X @ w + b)
    g_an = -(X.T @ (y - p)) / len(y)
    h = 1e-6
    g_num = np.array([(bce(y, sigmoid(X @ (w + h * e) + b)) - bce(y, sigmoid(X @ (w - h * e) + b))) / (2 * h)
                      for e in np.eye(4)])
    err = np.max(np.abs(g_an - g_num))
    assert err < 1e-7, err
    print(f"OK  gradient: max różnica analityczny vs numeryczny = {err:.1e}")


def test_metrics_vs_sklearn():
    from sklearn.metrics import f1_score, precision_score, recall_score
    rng = np.random.default_rng(1)
    y = (rng.random(200) > 0.4).astype(float); p = rng.random(200)
    m = metrics(y, p); yhat = (p >= 0.5).astype(float)
    assert abs(m["precision"] - precision_score(y, yhat)) < 1e-12
    assert abs(m["recall"] - recall_score(y, yhat)) < 1e-12
    assert abs(m["f1"] - f1_score(y, yhat)) < 1e-12
    print("OK  metryki: precision/recall/F1 zgodne ze scikit-learn")


def test_loss_decreases():
    (Xtr, ytr), _, _, _ = load_data()
    h = LogisticRegressionGD(lr=0.1, epochs=500).fit(Xtr, ytr).history["loss"]
    assert abs(h[0] - np.log(2)) < 1e-12, "przy w=0 koszt = ln 2"
    assert np.all(np.diff(h) <= 1e-12), "przy małym alfa koszt nie może rosnąć"
    print(f"OK  uczenie: koszt start ln2={h[0]:.4f}, monotonicznie maleje do {h[-1]:.4f}")


def test_same_minimum_as_sklearn():
    from sklearn.linear_model import LogisticRegression
    (Xtr, ytr), _, _, _ = load_data()
    m = LogisticRegressionGD(lr=0.5, epochs=200_000, stop="tol", tol=1e-14).fit(Xtr, ytr)
    sk = LogisticRegression(C=np.inf, max_iter=100_000, tol=1e-12).fit(Xtr, ytr)
    dL = abs(bce(ytr, m.predict_proba(Xtr)) - bce(ytr, sk.predict_proba(Xtr)[:, 1]))
    assert dL < 1e-6, dL
    print(f"OK  minimum: koszt nasz vs sklearn różni się o {dL:.1e}")


if __name__ == "__main__":
    test_sigmoid()
    test_gradient()
    test_metrics_vs_sklearn()
    test_loss_decreases()
    test_same_minimum_as_sklearn()
    print("Wszystkie testy zaliczone.")

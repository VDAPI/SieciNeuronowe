# -*- coding: utf-8 -*-
"""
Testy poprawności propagacji wstecznej z mlp.py.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu.]

Uruchomienie:  python test_mlp.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mlp import MLP, Linear, ReLU, SigmoidBCE, bce, load_data, sigmoid  # noqa: E402


def numeric_grad(f, A, h=1e-6):
    """Gradient skalarnej funkcji f() po każdym elemencie tablicy A (różnice centralne, A zmieniana w miejscu)."""
    g = np.zeros_like(A)
    for i in np.ndindex(A.shape):
        old = A[i]
        A[i] = old + h; fp = f()
        A[i] = old - h; fm = f()
        A[i] = old
        g[i] = (fp - fm) / (2 * h)
    return g


def test_linear():
    """Linear.backward: dL/dW, dL/db i dL/dx dla L = sum(G * y) (G - dowolne "nadchodzące" dL/dy)."""
    rng = np.random.default_rng(0)
    lin = Linear(4, 3, rng, scale=1.0)
    X = rng.normal(size=(6, 4)); G = rng.normal(size=(6, 3))
    loss = lambda: float(np.sum(G * lin.forward(X)))  # noqa: E731
    lin.forward(X); dX = lin.backward(G)
    errs = [np.max(np.abs(lin.dW - numeric_grad(loss, lin.W))), np.max(np.abs(lin.db - numeric_grad(loss, lin.b))),
            np.max(np.abs(dX - numeric_grad(loss, X)))]
    assert max(errs) < 1e-7, errs
    print(f"OK  Linear: dL/dW, dL/db, dL/dx zgodne z numerycznymi (max różnica {max(errs):.1e})")


def test_relu():
    rng = np.random.default_rng(1)
    relu = ReLU()
    X = rng.normal(size=(5, 4)); X[np.abs(X) < 1e-3] = 0.5   # z dala od punktu nieróżniczkowalności 0
    G = rng.normal(size=(5, 4))
    loss = lambda: float(np.sum(G * relu.forward(X)))  # noqa: E731
    relu.forward(X); dX = relu.backward(G)
    err = np.max(np.abs(dX - numeric_grad(loss, X)))
    assert err < 1e-7, err
    assert np.all(dX[X < 0] == 0), "dla x < 0 gradient ReLU musi być zerowy"
    print(f"OK  ReLU: dL/dx = dL/dy * [x > 0] (max różnica {err:.1e})")


def test_sigmoid_bce():
    rng = np.random.default_rng(2)
    z = rng.normal(size=8); y = (rng.random(8) > 0.5).astype(float)
    head = SigmoidBCE()
    head.forward(z, y); dz = head.backward()[:, 0]
    err = np.max(np.abs(dz - numeric_grad(lambda: bce(y, sigmoid(z)), z)))
    assert err < 1e-8, err
    assert np.allclose(dz, (sigmoid(z) - y) / len(y)), "dL/dz = (p - y) / N"
    print(f"OK  sigmoida + BCE: dL/dz = (p - y)/N (max różnica {err:.1e})")


def test_whole_network():
    """Gradient całej sieci 5-6-4-1 (dwie warstwy ukryte) względem każdej wagi każdej warstwy."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 5)); y = (rng.random(20) > 0.5).astype(float)
    net = MLP(5, (6, 4), init_scale=0.5, seed=3)
    net.loss_and_grad(X, y)
    loss = lambda: bce(y, net.predict_proba(X))  # noqa: E731
    err = 0.0
    for layer in net.linears:
        err = max(err, np.max(np.abs(layer.dW - numeric_grad(loss, layer.W))),
                  np.max(np.abs(layer.db - numeric_grad(loss, layer.b))))
    assert err < 1e-7, err
    print(f"OK  cała sieć (2 warstwy ukryte): propagacja wsteczna = gradient numeryczny (max różnica {err:.1e})")


def test_no_hidden_layer_is_logistic_regression():
    """Bez warstw ukrytych sieć to regresja logistyczna z ćw. 2: gradient -(1/N) X^T (y - p)."""
    (Xtr, ytr), _, _, _ = load_data()
    net = MLP(Xtr.shape[1], (), seed=0)
    net.loss_and_grad(Xtr, ytr)
    p = net.predict_proba(Xtr)
    err = np.max(np.abs(net.linears[0].dW[0] + Xtr.T @ (ytr - p) / len(ytr)))
    assert err < 1e-12, err
    print(f"OK  0 warstw ukrytych = regresja logistyczna z ćw. 2 (różnica gradientu {err:.1e})")


def test_loss_decreases():
    (Xtr, ytr), (Xva, yva), _, _ = load_data()
    h = MLP(Xtr.shape[1], (17,), seed=0).fit(Xtr, ytr, Xva, yva, epochs=3000).history["loss"]
    assert abs(h[0] - np.log(2)) < 1e-3, "przy wagach ~0 sieć zwraca p ~ 0,5, koszt ~ ln 2"
    assert h[-1] < h[0] - 0.2, "sieć musi się uczyć"
    print(f"OK  uczenie: koszt {h[0]:.4f} -> {h[-1]:.4f} po 3000 epokach")


if __name__ == "__main__":
    test_linear()
    test_relu()
    test_sigmoid_bce()
    test_whole_network()
    test_no_hidden_layer_is_logistic_regression()
    test_loss_decreases()
    print("Wszystkie testy zaliczone.")

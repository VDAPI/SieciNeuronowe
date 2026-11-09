# -*- coding: utf-8 -*-
"""
Testy poprawności: ADAM, prosty gradient, tryby paczkowania, inicjalizacja, propagacja wsteczna.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu.]

Uruchomienie:  python test_optim.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from optim import GD, MLP, Adam, bce, load_data, train  # noqa: E402


def test_adam_matches_formulas():
    """Dwa kroki ADAM liczone ręcznie ze wzorów z treści."""
    b1, b2, lr, eps = 0.9, 0.999, 0.1, 1e-8
    w = np.array([1.0, -2.0]); g1 = np.array([0.5, -0.1]); g2 = np.array([0.2, 0.3])
    opt = Adam(lr, b1, b2, eps)
    opt.step([(w, g1)]); opt.step([(w, g2)])
    m = v = np.zeros(2); w_ref = np.array([1.0, -2.0])
    for t, g in enumerate([g1, g2], 1):
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        w_ref = w_ref - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    assert np.allclose(w, w_ref, atol=1e-12), (w, w_ref)
    print("OK  ADAM: dwa kroki zgodne ze wzorami (m, v, korekcja biasu)")


def test_adam_first_step_is_lr():
    """Po korekcji biasu pierwszy krok ADAM ma długość ~lr niezależnie od skali gradientu
    (dopóki gradient jest dużo większy od eps)."""
    for scale in [1e-3, 1.0, 1e6]:
        w = np.zeros(3); g = scale * np.array([1.0, -3.0, 0.5])
        Adam(0.01).step([(w, g)])
        assert np.allclose(np.abs(w), 0.01, rtol=1e-3), (scale, w)
    print("OK  ADAM: pierwszy krok = α · znak(g) dla gradientów od 1e-3 do 1e6")


def test_gd():
    w = np.array([1.0, 2.0]); GD(0.1).step([(w, np.array([10.0, -10.0]))])
    assert np.allclose(w, [0.0, 3.0])
    print("OK  GD: w <- w - α g")


def test_batch_modes_step_counts():
    (Xtr, ytr), (Xva, yva), _, _ = load_data()
    n = len(ytr)
    for mode, per_epoch in [("full", 1), ("minibatch", int(np.ceil(n / 32))), ("single", n)]:
        m = train(MLP(Xtr.shape[1], (8,), seed=0), GD(0.01), Xtr, ytr, Xva, yva, mode=mode, epochs=3)
        assert m.history["step"][-1] == 3 * per_epoch, (mode, m.history["step"][-1])
    print(f"OK  tryby: kroków na epokę pełny zbiór 1, paczki {int(np.ceil(n / 32))}, pojedyncze {n}")


def test_full_batch_gradient_is_mean_of_single():
    """Gradient na pełnym zbiorze = średnia gradientów pojedynczych przykładów (koszt to średnia)."""
    (Xtr, ytr), _, _, _ = load_data()
    net = MLP(Xtr.shape[1], (8, 8), seed=1)
    net.loss_and_grad(Xtr, ytr)
    full = net.linears[0].dW.copy()
    mean = np.zeros_like(full)
    for i in range(len(ytr)):
        net.loss_and_grad(Xtr[i:i + 1], ytr[i:i + 1])
        mean += net.linears[0].dW / len(ytr)
    assert np.allclose(full, mean, atol=1e-12)
    print("OK  gradient pełnego zbioru = średnia gradientów pojedynczych przykładów")


def test_he_init_keeps_variance():
    """He: odchylenie wag sqrt(2 / d_in) i wariancja sygnału w głąb sieci ReLU ~ stała (vs N(0; 1): rośnie)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 64))
    stds = {}
    for name, kind, sc in [("he", "he", 1.0), ("n1", "normal", 1.0)]:
        net = MLP(64, (64,) * 6, kind, sc, seed=0)
        A = X
        for layer in net.layers[:-1]:
            A = layer.forward(A)
        stds[name] = float(np.std(A))
    assert abs(np.std(MLP(64, (64,), "he", seed=0).linears[0].W) - np.sqrt(2 / 64)) < 0.01
    assert 0.3 < stds["he"] < 3, stds
    assert stds["n1"] > 1e3, stds
    print(f"OK  He: std po 6 warstwach {stds['he']:.2f} (N(0; 1): {stds['n1']:.1e})")


def test_backprop_numeric():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 5)); y = (rng.random(20) > 0.5).astype(float)
    net = MLP(5, (6, 4), "normal", 0.5, seed=3)
    net.loss_and_grad(X, y)
    err, h = 0.0, 1e-6
    for layer in net.linears:
        for A, G in [(layer.W, layer.dW), (layer.b, layer.db)]:
            for i in np.ndindex(A.shape):
                old = A[i]
                A[i] = old + h; fp = bce(y, net.predict_proba(X))
                A[i] = old - h; fm = bce(y, net.predict_proba(X))
                A[i] = old
                err = max(err, abs(G[i] - (fp - fm) / (2 * h)))
    assert err < 1e-7, err
    print(f"OK  propagacja wsteczna = gradient numeryczny (max różnica {err:.1e})")


if __name__ == "__main__":
    test_adam_matches_formulas()
    test_adam_first_step_is_lr()
    test_gd()
    test_batch_modes_step_counts()
    test_full_batch_gradient_is_mean_of_single()
    test_he_init_keeps_variance()
    test_backprop_numeric()
    print("Wszystkie testy zaliczone.")

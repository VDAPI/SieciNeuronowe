# -*- coding: utf-8 -*-
"""
SIECI NEURONOWE - ćwiczenie 3: wielowarstwowa sieć neuronowa (MLP) uczona propagacją wsteczną.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu należy go
 oznaczyć, zrozumieć i umieć samodzielnie wyjaśnić.]

Sieć = sekwencja modułów, każdy ma forward() i backward():
  Linear      y = W x + b          backward: dL/dW = (dL/dy)^T x,  dL/db = dL/dy,  dL/dx = dL/dy W
  ReLU        y = max(0, x)        backward: dL/dx = dL/dy * [x > 0]   (macierz diagonalna -> mnożenie element-wise)
  SigmoidBCE  p = sigma(z), L = BCE(y, p)   backward: dL/dz = (p - y) / N   (jak w regresji logistycznej)

Konwencja: wiersz macierzy X to jeden przykład (x^T), więc paczka N przykładów to macierz (N, dim).
dL/dy dla jednego przykładu to wektor wierszowy (1, dim_y) - w paczce (N, dim_y), czyli dokładnie
konwencja z treści ćwiczenia, tylko wiele przykładów naraz.

Uczenie: pełny batch gradient descent, w <- w - alpha * dL/dw, wagi z N(0; 0,01), alpha = 0,01.

Uruchomienie:  python mlp.py      (wyniki -> figures/ oraz results.json)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 42
LR = 0.01            # współczynnik uczenia z treści ćwiczenia
INIT_SCALE = 0.01    # odchylenie standardowe wag początkowych z treści ćwiczenia
RESULTS = {}


# ----------------------------------------------------------------------------
# 1. Dane
# ----------------------------------------------------------------------------
sys.path.insert(0, HERE)
from data import load_splits  # noqa: E402  (preprocessing taki sam jak w ćwiczeniach 1-2)


def load_data(scaling="standard", seed=SEED):
    """scaling: 'standard' (średnia 0, odchylenie 1), 'minmax' (zakres [0, 1]), 'none' (surowe wartości).
    Statystyki skalowania zawsze ze zbioru treningowego."""
    if scaling == "standard":
        return load_splits(seed=seed, standardize=True)
    (Xtr, ytr), (Xva, yva), (Xte, yte), names = load_splits(seed=seed, standardize=False)
    if scaling == "minmax":
        lo, hi = Xtr.min(0), Xtr.max(0)
        span = np.where(hi > lo, hi - lo, 1.0)
        Xtr, Xva, Xte = [(X - lo) / span for X in (Xtr, Xva, Xte)]
    return (Xtr, ytr), (Xva, yva), (Xte, yte), names


# ----------------------------------------------------------------------------
# 2. Funkcje pomocnicze (jak w ćw. 2)
# ----------------------------------------------------------------------------
def sigmoid(z):
    # stabilna numerycznie wersja (brak overflow w exp dla dużych |z|)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(float)
    tp = np.sum((yhat == 1) & (y == 1)); fp = np.sum((yhat == 1) & (y == 0))
    fn = np.sum((yhat == 0) & (y == 1)); tn = np.sum((yhat == 0) & (y == 0))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(y)
    return dict(precision=float(prec), recall=float(rec), f1=float(f1), accuracy=float(acc),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


# ----------------------------------------------------------------------------
# 3. Moduły: forward zapamiętuje wejście, backward liczy gradient wag i zwraca dL/d(wejście)
# ----------------------------------------------------------------------------
class Linear:
    """y = W x + b, W: (n_out, n_in). Wagi z N(0; scale) albo He: N(0; sqrt(2 / n_in))."""

    def __init__(self, n_in, n_out, rng, scale=INIT_SCALE, init="normal"):
        std = np.sqrt(2.0 / n_in) if init == "he" else scale
        self.W = rng.normal(0.0, std, (n_out, n_in))
        self.b = np.zeros(n_out)

    def forward(self, X):
        self.X = X                         # potrzebne w backward do dL/dW
        return X @ self.W.T + self.b

    def backward(self, dY):                # dY = dL/dy, kształt (N, n_out)
        self.dW = dY.T @ self.X            # dL/dW, kształt (n_out, n_in) - suma po przykładach
        self.db = dY.sum(0)                # dL/db
        return dY @ self.W                 # dL/dx, kształt (N, n_in)


class ReLU:
    def forward(self, X):
        self.mask = X > 0                  # pochodna ReLU: 1 dla x > 0, 0 w przeciwnym razie
        return X * self.mask

    def backward(self, dY):
        # dy/dx to macierz diagonalna z maską na przekątnej -> zamiast mnożenia macierzy mnożenie element-wise
        return dY * self.mask


class SigmoidBCE:
    """Wyjście sieci (sigmoida) połączone z entropią krzyżową - ta sama funkcja kosztu co w regresji logistycznej.
    Razem, bo pochodna upraszcza się do (p - y) / N: stabilnie numerycznie także przy nasyceniu sigmoidy."""

    def forward(self, z, y):
        self.p, self.y = sigmoid(z), y
        return bce(y, self.p)

    def backward(self):
        return ((self.p - self.y) / len(self.y))[:, None]     # dL/dz, kształt (N, 1)


# ----------------------------------------------------------------------------
# 4. Sieć: Linear -> ReLU -> ... -> Linear -> SigmoidBCE
# ----------------------------------------------------------------------------
class MLP:
    def __init__(self, n_in, hidden=(17,), init="normal", init_scale=INIT_SCALE, seed=SEED):
        """hidden: liczby neuronów kolejnych warstw ukrytych; () = brak warstw ukrytych (regresja logistyczna)."""
        rng = np.random.default_rng(seed)
        self.layers = []
        prev = n_in
        for h in hidden:
            self.layers += [Linear(prev, h, rng, init_scale, init), ReLU()]
            prev = h
        self.layers.append(Linear(prev, 1, rng, init_scale, init))
        self.linears = [layer for layer in self.layers if isinstance(layer, Linear)]
        self.loss_fn = SigmoidBCE()

    def forward(self, X):
        for layer in self.layers:          # przejście w przód: w kolejności operacji
            X = layer.forward(X)
        return X[:, 0]                     # logit z

    def predict_proba(self, X):
        return sigmoid(self.forward(X))

    def loss_and_grad(self, X, y):
        loss = self.loss_fn.forward(self.forward(X), y)
        d = self.loss_fn.backward()
        for layer in reversed(self.layers):    # przejście wstecz: w odwrotnej kolejności
            d = layer.backward(d)
        return loss

    def step(self, lr):
        for layer in self.linears:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    def get_params(self):
        return [(layer.W.copy(), layer.b.copy()) for layer in self.linears]

    def set_params(self, params):
        for layer, (W, b) in zip(self.linears, params):
            layer.W, layer.b = W.copy(), b.copy()

    def fit(self, X, y, Xv=None, yv=None, lr=LR, epochs=1000, log_every=10):
        """Pełny batch GD przez `epochs` epok. Koszt train/val co epokę, metryki i normy gradientu co
        `log_every` epok. Zapamiętuje wagi z najmniejszym kosztem walidacyjnym (best_epoch)."""
        h = {k: [] for k in ["loss", "val_loss", "log_epoch", "f1", "val_f1", "val_precision", "val_recall",
                             "grad_norms"]}
        best = (np.inf, None, 0)
        for ep in range(epochs):
            loss = self.loss_and_grad(X, y)           # koszt i gradienty dla wag sprzed kroku
            h["loss"].append(loss)
            if Xv is not None:
                pv = self.predict_proba(Xv)
                h["val_loss"].append(bce(yv, pv))
                if h["val_loss"][-1] < best[0] - 1e-9:
                    best = (h["val_loss"][-1], self.get_params(), ep)
            if ep % log_every == 0 or ep == epochs - 1:
                h["log_epoch"].append(ep)
                h["f1"].append(metrics(y, self.loss_fn.p)["f1"])
                h["grad_norms"].append([float(np.sqrt(np.sum(l.dW ** 2) + np.sum(l.db ** 2))) for l in self.linears])
                if Xv is not None:
                    mv = metrics(yv, pv)
                    h["val_f1"].append(mv["f1"]); h["val_precision"].append(mv["precision"])
                    h["val_recall"].append(mv["recall"])
            if not np.isfinite(loss):
                break
            self.step(lr)
        self.history = h
        self.best_val_loss, self.best_params, self.best_epoch = best
        return self


def epochs_to(loss_hist, target):
    """Pierwsza epoka, w której koszt spadł do poziomu target (None, jeśli nigdy)."""
    idx = np.where(np.asarray(loss_hist) <= target)[0]
    return int(idx[0]) if len(idx) else None


PLATEAU = 0.6        # koszt stałej odpowiedzi (odsetek chorych) to ok. 0,69; "wyjście z plateau" = koszt < 0,6


def summarize(m, data):
    """Liczby jednego przebiegu: koszty, epoki, metryki dla wag końcowych i dla wag najlepszych na walidacji."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    L = m.history["loss"]
    final = dict(train=metrics(ytr, m.predict_proba(Xtr)), test=metrics(yte, m.predict_proba(Xte)))
    last = m.get_params()
    m.set_params(m.best_params)
    early = dict(val=metrics(yva, m.predict_proba(Xva)), test=metrics(yte, m.predict_proba(Xte)))
    m.set_params(last)
    return dict(final_train_loss=float(L[-1]), min_train_loss=float(np.nanmin(L)),
                final_val_loss=float(m.history["val_loss"][-1]), best_val_loss=float(m.best_val_loss),
                best_epoch=int(m.best_epoch), plateau_exit=epochs_to(L, PLATEAU),
                diverged=bool(not np.isfinite(L[-1])), final=final, early_stopping=early)


def mean_of(items, key_fn):
    vals = [v for v in (key_fn(item) for item in items) if v is not None]
    return float(np.mean(vals)) if vals else None


# ----------------------------------------------------------------------------
# 5. Eksperymenty
# ----------------------------------------------------------------------------
E = 30_000           # epok w eksperymentach z szerokością i skalowaniem
E_DEPTH = 150_000    # głębsze sieci przy wagach N(0; 0,01) bardzo długo stoją w miejscu
SEEDS = (0, 1, 2)
SCALINGS = [("none", "surowe"), ("minmax", "min-max [0, 1]"), ("standard", "standaryzacja")]


def main():
    (Xtr, ytr), (Xva, yva), (Xte, yte), names = load_data("standard")
    d = Xtr.shape[1]
    D = ((Xtr, ytr), (Xva, yva), (Xte, yte))
    print(f"train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}; cech: {d}")
    RESULTS["data"] = dict(n_train=len(ytr), n_val=len(yva), n_test=len(yte), n_features=d,
                           pos_rate_train=float(ytr.mean()),
                           constant_loss=bce(ytr, np.full_like(ytr, ytr.mean())))
    RESULTS["settings"] = dict(lr=LR, init_scale=INIT_SCALE, epochs=E, epochs_depth=E_DEPTH, seeds=list(SEEDS))

    # ------------------------------------------------------------------ 5.1 Model bazowy: 1 warstwa ukryta, 17 neuronów
    base = MLP(d, (d,), seed=0).fit(Xtr, ytr, Xva, yva, epochs=E)
    h = base.history
    RESULTS["baseline"] = summarize(base, D)
    print("[baseline]", {k: v for k, v in RESULTS["baseline"].items() if not isinstance(v, dict)})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="walidacja")
    ax[0].axvline(base.best_epoch, c="tab:red", ls="--", label=f"min. kosztu walidacyjnego ({base.best_epoch})")
    ax[0].set_xscale("log"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("entropia krzyżowa"); ax[0].legend()
    ax[0].set_title("MLP 17-17-1: koszt")
    for k, ls in [("val_f1", "-"), ("val_precision", "--"), ("val_recall", ":")]:
        ax[1].plot(h["log_epoch"], h[k], ls, label=k)
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1); ax[1].set_xlabel("epoka"); ax[1].legend()
    ax[1].set_title("Metryki (walidacja)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "1_baseline_learning_curve.png"), dpi=130); plt.close()

    # ------------------------------------------------------------------ 5.2 Szerokość warstwy ukrytej
    widths = [2, 4, d, 4 * d, 256]
    width_res, curves = {}, {}
    for w in widths:
        runs = [MLP(d, (w,), seed=s).fit(Xtr, ytr, Xva, yva, epochs=E, log_every=100) for s in SEEDS]
        sums = [summarize(r, D) for r in runs]
        width_res[str(w)] = dict(
            runs=sums,
            plateau_exit=mean_of(sums, lambda s: s["plateau_exit"]),
            final_train_loss=mean_of(sums, lambda s: s["final_train_loss"]),
            best_val_loss=mean_of(sums, lambda s: s["best_val_loss"]),
            best_epoch=mean_of(sums, lambda s: s["best_epoch"]),
            final_val_loss=mean_of(sums, lambda s: s["final_val_loss"]),
            test_f1_early=mean_of(sums, lambda s: s["early_stopping"]["test"]["f1"]),
            test_f1_final=mean_of(sums, lambda s: s["final"]["test"]["f1"]))
        curves[w] = (np.mean([r.history["loss"] for r in runs], 0), np.mean([r.history["val_loss"] for r in runs], 0))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for w in widths:
        ax[0].plot(curves[w][0], label=f"{w} neuronów"); ax[1].plot(curves[w][1], label=f"{w} neuronów")
    for a, t in zip(ax, ["koszt treningowy", "koszt walidacyjny"]):
        a.set_xscale("log"); a.set_xlabel("epoka"); a.set_ylabel(t); a.legend(fontsize=8)
    ax[0].set_yscale("log"); ax[0].set_title("Szerokość warstwy ukrytej: train")
    ax[1].set_ylim(0.3, 1.5); ax[1].set_title("Szerokość warstwy ukrytej: walidacja")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "2_hidden_width.png"), dpi=130); plt.close()
    RESULTS["hidden_width"] = width_res
    print("[width]")
    for k, v in width_res.items():
        print(f"   {k:<4}", {kk: vv for kk, vv in v.items() if kk != "runs"})

    # ------------------------------------------------------------------ 5.3 Dane znormalizowane i nieznormalizowane
    scal_res = {}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for sc, label in SCALINGS:
        (Xa, ya), (Xb, yb), (Xc, yc), _ = load_data(sc)
        scal_res[sc] = {"cond_XtX": float(np.linalg.cond(Xa.T @ Xa))}
        for arch, hidden in [("logreg", ()), ("mlp", (d,))]:
            m = MLP(d, hidden, seed=0).fit(Xa, ya, Xb, yb, epochs=E, log_every=100)
            scal_res[sc][arch] = summarize(m, ((Xa, ya), (Xb, yb), (Xc, yc)))
            ax[0 if arch == "mlp" else 1].plot(m.history["loss"], label=label)
    ax[0].set_title("MLP 17-17-1"); ax[1].set_title("Bez warstwy ukrytej (regresja logistyczna)")
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("epoka"); a.set_ylabel("koszt treningowy"); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "3_scaling.png"), dpi=130); plt.close()
    RESULTS["scaling"] = scal_res
    print("[scaling]")
    for k, v in scal_res.items():
        print(f"   {k:<9} cond={v['cond_XtX']:.2e}", " ".join(
            f"{a}: L={v[a]['final_train_loss']:.3f} F1es={v[a]['early_stopping']['test']['f1']:.3f}"
            for a in ("logreg", "mlp")))

    # ------------------------------------------------------------------ 5.4 Liczba warstw ukrytych
    depths = [0, 1, 2, 3, 4]
    depth_res, dcurves, grad0 = {}, {}, {}
    for n in depths:
        runs = [MLP(d, (d,) * n, seed=s).fit(Xtr, ytr, Xva, yva, epochs=E_DEPTH, log_every=1000) for s in SEEDS]
        sums = [summarize(r, D) for r in runs]
        depth_res[str(n)] = dict(
            runs=sums,
            plateau_exit=[s["plateau_exit"] for s in sums],
            final_train_loss=mean_of(sums, lambda s: s["final_train_loss"]),
            best_val_loss=mean_of(sums, lambda s: s["best_val_loss"]),
            test_f1_early=mean_of(sums, lambda s: s["early_stopping"]["test"]["f1"]),
            grad_norms_start=runs[0].history["grad_norms"][0])
        dcurves[n] = runs[0].history["loss"]
        grad0[n] = runs[0].history["grad_norms"][0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for n in depths:
        ax[0].plot(dcurves[n], label=f"{n} warstw ukrytych")
        ax[1].plot(range(1, len(grad0[n]) + 1), grad0[n], marker="o", label=f"{n} warstw ukrytych")
    ax[0].set_xscale("log"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("koszt treningowy"); ax[0].legend(fontsize=8)
    ax[0].set_title("Liczba warstw (17 neuronów, N(0; 0,01), α = 0,01)")
    ax[1].set_yscale("log"); ax[1].set_xlabel("warstwa liniowa (1 = przy wejściu)"); ax[1].set_ylabel("||dL/dW|| w epoce 0")
    ax[1].set_title("Gradient maleje w stronę wejścia"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "4_depth.png"), dpi=130); plt.close()
    RESULTS["depth"] = depth_res
    print("[depth]")
    for k, v in depth_res.items():
        print(f"   {k}", {kk: vv for kk, vv in v.items() if kk != "runs"})

    # ------------------------------------------------------------------ 5.5 Dodatkowo: inicjalizacja He w głębszych sieciach
    he_res = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in [3, 4]:
        for init, ls in [("normal", "-"), ("he", "--")]:
            m = MLP(d, (d,) * n, init=init, seed=0).fit(Xtr, ytr, Xva, yva, epochs=E, log_every=100)
            he_res[f"{n}_{init}"] = summarize(m, D)
            ax.plot(m.history["loss"], ls, label=f"{n} warstwy, " + ("N(0; 0,01)" if init == "normal" else "He"))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("epoka"); ax.set_ylabel("koszt treningowy")
    ax.set_title("Inicjalizacja He a N(0; 0,01)"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "5_he_init.png"), dpi=130); plt.close()
    RESULTS["he_init"] = he_res
    print("[he]")
    for k, v in he_res.items():
        print(f"   {k:<9} exit={v['plateau_exit']} bestval={v['best_val_loss']:.3f}@{v['best_epoch']} "
              f"F1es={v['early_stopping']['test']['f1']:.3f}")

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

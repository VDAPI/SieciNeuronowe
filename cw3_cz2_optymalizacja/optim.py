# -*- coding: utf-8 -*-
"""
SIECI NEURONOWE - ćwiczenie 3, cz. 2: optymalizacja sieci wielowarstwowej.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu należy go
 oznaczyć, zrozumieć i umieć samodzielnie wyjaśnić.]

Sieć z cz. 1 (moduły Linear / ReLU / SigmoidBCE z forward i backward), do tego:
  - tryby uczenia: pełny zbiór (batch), paczki (mini-batch), pojedyncze przykłady (SGD),
  - optymalizatory: prosty gradient  w <- w - alpha g
                    ADAM  m <- b1 m + (1-b1) g,  v <- b2 v + (1-b2) g^2,
                          m^ = m / (1-b1^t),  v^ = v / (1-b2^t),  w <- w - alpha m^ / (sqrt(v^) + eps)
  - inicjalizacja wag: N(0; sigma) albo He: N(0; sqrt(2 / d_in)).

Uruchomienie:  python optim.py      (wyniki -> figures/ oraz results.json)
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 42
RESULTS = {}


# ----------------------------------------------------------------------------
# 1. Dane
# ----------------------------------------------------------------------------
sys.path.insert(0, HERE)
from data import load_splits  # noqa: E402  (preprocessing taki sam jak w ćwiczeniach 1-3)


def load_data(seed=SEED):
    return load_splits(seed=seed, standardize=True)


# ----------------------------------------------------------------------------
# 2. Funkcje pomocnicze (jak w ćw. 2 i 3)
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
# 3. Moduły sieci (z cz. 1)
# ----------------------------------------------------------------------------
class Linear:
    """y = W x + b, W: (n_out, n_in). init='he': N(0; sqrt(2 / n_in)), inaczej N(0; scale)."""

    def __init__(self, n_in, n_out, rng, init="he", scale=1.0):
        std = np.sqrt(2.0 / n_in) if init == "he" else scale
        self.W = rng.normal(0.0, std, (n_out, n_in))
        self.b = np.zeros(n_out)

    def forward(self, X):
        self.X = X
        return X @ self.W.T + self.b

    def backward(self, dY):                # dY = dL/dy, (N, n_out)
        self.dW = dY.T @ self.X            # dL/dW
        self.db = dY.sum(0)                # dL/db
        return dY @ self.W                 # dL/dx


class ReLU:
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask

    def backward(self, dY):
        return dY * self.mask


class SigmoidBCE:
    """Sigmoida + entropia krzyżowa; dL/dz = (p - y) / N."""

    def forward(self, z, y):
        self.p, self.y = sigmoid(z), y
        return bce(y, self.p)

    def backward(self):
        return ((self.p - self.y) / len(self.y))[:, None]


class MLP:
    def __init__(self, n_in, hidden, init="he", scale=1.0, seed=SEED):
        rng = np.random.default_rng(seed)
        self.layers, prev = [], n_in
        for h in hidden:
            self.layers += [Linear(prev, h, rng, init, scale), ReLU()]
            prev = h
        self.layers.append(Linear(prev, 1, rng, init, scale))
        self.linears = [layer for layer in self.layers if isinstance(layer, Linear)]
        self.loss_fn = SigmoidBCE()

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X[:, 0]

    def predict_proba(self, X):
        return sigmoid(self.forward(X))

    def loss_and_grad(self, X, y):
        loss = self.loss_fn.forward(self.forward(X), y)
        d = self.loss_fn.backward()
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return loss

    def params_and_grads(self):
        """Pary (parametr, gradient); optymalizator zmienia parametry w miejscu."""
        out = []
        for layer in self.linears:
            out += [(layer.W, layer.dW), (layer.b, layer.db)]
        return out

    def get_params(self):
        return [(layer.W.copy(), layer.b.copy()) for layer in self.linears]

    def set_params(self, params):
        for layer, (W, b) in zip(self.linears, params):
            layer.W, layer.b = W.copy(), b.copy()


# ----------------------------------------------------------------------------
# 4. Optymalizatory
# ----------------------------------------------------------------------------
class GD:
    """Prosty spadek gradientu: w <- w - lr * g."""

    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, pairs):
        for w, g in pairs:
            w -= self.lr * g


class Adam:
    """ADAM: średnia krocząca gradientu (momentum) i jego kwadratu (skalowanie kroku), osobno dla każdej wagi."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = None, None, 0

    def step(self, pairs):
        if self.m is None:                                  # m i v startują od zera
            self.m = [np.zeros_like(w) for w, _ in pairs]
            self.v = [np.zeros_like(w) for w, _ in pairs]
        self.t += 1
        for (w, g), m, v in zip(pairs, self.m, self.v):
            m *= self.beta1; m += (1 - self.beta1) * g      # m_t = b1 m_(t-1) + (1 - b1) g
            v *= self.beta2; v += (1 - self.beta2) * g * g  # v_t = b2 v_(t-1) + (1 - b2) g^2
            m_hat = m / (1 - self.beta1 ** self.t)          # korekcja biasu (start od zera)
            v_hat = v / (1 - self.beta2 ** self.t)
            w -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ----------------------------------------------------------------------------
# 5. Uczenie w trzech trybach
# ----------------------------------------------------------------------------
def batch_size_for(mode, n, batch=32):
    return {"full": n, "minibatch": batch, "single": 1}[mode]


def train(model, opt, X, y, Xv, yv, mode="full", batch=32, epochs=500, seed=SEED, log_every=1):
    """Jedna epoka = jedno przejście przez zbiór treningowy w losowej kolejności, podzielony na paczki
    (full: 1 paczka = 1 krok, minibatch: ceil(N / batch) kroków, single: N kroków).
    Koszt train/val liczony na całych zbiorach po każdej epoce (dla porównywalności trybów)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    bs = batch_size_for(mode, n, batch)
    h = {k: [] for k in ["epoch", "step", "loss", "val_loss", "val_f1", "grad_norms", "time"]}
    best = (np.inf, None, 0)
    steps, t0 = 0, time.perf_counter()

    def log(ep):
        loss = model.loss_and_grad(X, y)                    # pełny gradient tylko do pomiaru
        pv = model.predict_proba(Xv)
        h["epoch"].append(ep); h["step"].append(steps); h["loss"].append(loss)
        h["val_loss"].append(bce(yv, pv)); h["val_f1"].append(metrics(yv, pv)["f1"])
        h["grad_norms"].append([float(np.sqrt(np.sum(l.dW ** 2))) for l in model.linears])
        h["time"].append(time.perf_counter() - t0)
        return loss

    log(0)
    for ep in range(1, epochs + 1):
        order = rng.permutation(n) if bs < n else np.arange(n)
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            model.loss_and_grad(X[idx], y[idx])
            opt.step(model.params_and_grads())
            steps += 1
        if ep % log_every == 0 or ep == epochs:
            loss = log(ep)
            if h["val_loss"][-1] < best[0] - 1e-9:
                best = (h["val_loss"][-1], model.get_params(), ep)
            if not np.isfinite(loss):
                break
    model.history = h
    model.best_val_loss, model.best_params, model.best_epoch = best
    return model


def epochs_to(hist, key, target):
    """Pierwsza zalogowana epoka, w której hist[key] <= target (None, jeśli nigdy)."""
    for ep, value in zip(hist["epoch"], hist[key]):
        if value <= target:
            return int(ep)
    return None


def summarize(m, data):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    h = m.history
    out = dict(start_loss=float(h["loss"][0]), final_loss=float(h["loss"][-1]),
               min_loss=float(np.nanmin(h["loss"])), diverged=bool(not np.isfinite(h["loss"][-1])),
               epochs_run=int(h["epoch"][-1]), steps=int(h["step"][-1]), time_s=float(h["time"][-1]),
               plateau_exit=epochs_to(h, "loss", PLATEAU), epochs_to_target=epochs_to(h, "loss", TARGET),
               best_val_loss=float(m.best_val_loss), best_epoch=int(m.best_epoch), test_f1_early=None)
    if m.best_params is not None:
        last = m.get_params()
        m.set_params(m.best_params)
        out["test_early"] = metrics(yte, m.predict_proba(Xte))
        out["test_f1_early"] = out["test_early"]["f1"]
        m.set_params(last)
    if out["epochs_to_target"] is not None:
        k = h["epoch"].index(out["epochs_to_target"])
        out["steps_to_target"], out["time_to_target"] = int(h["step"][k]), float(h["time"][k])
    return out


def mean_of(items, key):
    vals = [item[key] for item in items if item.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def activation_stds(model, X):
    """Odchylenie standardowe wyjścia każdej warstwy liniowej przy przejściu w przód (na starcie uczenia)."""
    stds, A = [], X
    for layer in model.layers:
        A = layer.forward(A)
        if isinstance(layer, Linear):
            stds.append(float(np.std(A)))
    return stds


# ----------------------------------------------------------------------------
# 6. Eksperymenty
# ----------------------------------------------------------------------------
NET = (17,) * 5      # 5 warstw ukrytych po 17 neuronów: dość głęboko, żeby problemy było widać
SEEDS = (0, 1, 2)
PLATEAU = 0.6        # koszt stałej odpowiedzi ~0,69; "wyjście z plateau" = koszt treningowy < 0,6
TARGET = 0.3         # "dobrze dopasowany" model: koszt treningowy <= 0,3
E_MODE, E_INIT, E_OPT = 1000, 500, 3000
INITS = [("N(0; 0,01)", "normal", 0.01), ("N(0; 1)", "normal", 1.0), ("He", "he", 1.0),
         ("N(0; 10), zepsuta", "normal", 10.0)]
OPTS = [("GD α = 0,01", lambda: GD(0.01)), ("ADAM α = 0,001", lambda: Adam(0.001)),
        ("ADAM α = 0,01", lambda: Adam(0.01))]


def saddle_f(p):
    x, y = p
    return x * x + y ** 4 / 4 - y * y / 2      # punkt siodłowy w (0, 0), minima w (0, ±1)


def saddle_grad(p):
    x, y = p
    return np.array([2 * x, y ** 3 - y])


def run_saddle(opt, start=(1.0, 1e-4), steps=150):
    p = np.array(start, dtype=float)
    traj = [p.copy()]
    for _ in range(steps):
        opt.step([(p, saddle_grad(p))])
        traj.append(p.copy())
    return np.array(traj)


def main():
    (Xtr, ytr), (Xva, yva), (Xte, yte), names = load_data()
    d = Xtr.shape[1]
    D = ((Xtr, ytr), (Xva, yva), (Xte, yte))
    print(f"train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}; sieć {d}-{'-'.join(map(str, NET))}-1")
    RESULTS["data"] = dict(n_train=len(ytr), n_val=len(yva), n_test=len(yte), n_features=d)
    RESULTS["settings"] = dict(hidden=list(NET), seeds=list(SEEDS), plateau=PLATEAU, target=TARGET,
                               epochs_modes=E_MODE, epochs_init=E_INIT, epochs_opt=E_OPT)

    # ------------------------------------------------------------------ 6.1 Punkt siodłowy na płaszczyźnie
    trajs = {"GD (α = 0,1)": run_saddle(GD(0.1)), "ADAM (α = 0,1)": run_saddle(Adam(0.1))}
    RESULTS["saddle"] = {k: dict(escape_step=int(np.argmax(np.abs(t[:, 1]) > 0.5)), final=t[-1].tolist(),
                                 final_f=float(saddle_f(t[-1]))) for k, t in trajs.items()}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    xs, ys = np.meshgrid(np.linspace(-1.2, 1.2, 200), np.linspace(-1.5, 1.5, 200))
    ax[0].contour(xs, ys, saddle_f((xs, ys)), levels=30, cmap="Greys", linewidths=0.6)
    for k, t in trajs.items():
        ax[0].plot(t[:, 0], t[:, 1], ".-", ms=3, label=k)
        ax[1].plot(np.abs(t[:, 1]), label=k)
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].legend(); ax[0].set_title("f(x, y) = x² + y⁴/4 − y²/2")
    ax[1].set_yscale("log"); ax[1].set_xlabel("krok"); ax[1].set_ylabel("|y|"); ax[1].legend()
    ax[1].set_title("Ucieczka z punktu siodłowego")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "1_saddle.png"), dpi=130); plt.close()
    print("[saddle]", RESULTS["saddle"])

    # ------------------------------------------------------------------ 6.2 Pełny zbiór vs paczki vs pojedyncze przykłady
    mode_res, mode_runs = {}, {}
    for mode in ["full", "minibatch", "single"]:
        runs = [train(MLP(d, NET, "he", seed=s), GD(0.01), Xtr, ytr, Xva, yva, mode=mode, epochs=E_MODE, seed=s)
                for s in SEEDS]
        sums = [summarize(r, D) for r in runs]
        mode_runs[mode] = runs
        mode_res[mode] = dict(runs=sums, steps_per_epoch=sums[0]["steps"] // E_MODE,
                              **{k: mean_of(sums, k) for k in ["epochs_to_target", "steps_to_target", "time_to_target",
                                                                "best_val_loss", "best_epoch", "test_f1_early",
                                                                "final_loss", "time_s"]})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for mode, label in [("full", "pełny zbiór"), ("minibatch", "paczki po 32"), ("single", "pojedyncze przykłady")]:
        h = mode_runs[mode][0].history
        ax[0].plot(h["epoch"], h["loss"], label=label)
        ax[1].plot(np.maximum(h["step"], 1), h["loss"], label=label)
        ax[2].plot(h["epoch"], h["val_loss"], label=label)
    for a, xl, t in zip(ax, ["epoka", "liczba kroków (aktualizacji wag)", "epoka"],
                        ["Koszt treningowy / epoka", "Koszt treningowy / krok", "Koszt walidacyjny"]):
        a.set_xscale("symlog"); a.set_xlabel(xl); a.set_title(t); a.legend(fontsize=8)
    ax[0].set_yscale("log"); ax[1].set_yscale("log"); ax[2].set_ylim(0.3, 2)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "2_batch_modes.png"), dpi=130); plt.close()
    RESULTS["batch_modes"] = mode_res
    print("[modes]")
    for k, v in mode_res.items():
        print(f"   {k:<10}", {kk: vv for kk, vv in v.items() if kk != "runs"})

    # ------------------------------------------------------------------ 6.3 Inicjalizacja wag
    init_res = {}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for label, kind, sc in INITS:
        models = [MLP(d, NET, kind, sc, seed=s) for s in SEEDS]
        with np.errstate(all="ignore"):
            stds = activation_stds(models[0], Xtr)
            models[0].loss_and_grad(Xtr, ytr)
            grads0 = [float(np.sqrt(np.sum(l.dW ** 2))) for l in models[0].linears]
            runs = [train(m, GD(0.01), Xtr, ytr, Xva, yva, mode="minibatch", epochs=E_INIT, seed=s)
                    for m, s in zip(models, SEEDS)]
        sums = [summarize(r, D) for r in runs]
        init_res[label] = dict(runs=sums, activation_std=stds, grad_norms_start=grads0,
                               n_diverged=int(sum(s["diverged"] for s in sums)),
                               **{k: mean_of(sums, k) for k in ["start_loss", "final_loss", "plateau_exit",
                                                                "epochs_to_target", "best_val_loss", "test_f1_early"]})
        h = runs[0].history
        ax[0].plot(h["epoch"], h["loss"], marker="x" if sums[0]["diverged"] else None, label=label)
        ax[1].plot(range(1, len(stds) + 1), stds, marker="o", label=label)
        ax[2].plot(range(1, len(grads0) + 1), grads0, marker="o", label=label)
    ax[0].set_xscale("symlog"); ax[0].set_yscale("log"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("koszt treningowy")
    ax[0].set_title("Koszt (paczki po 32, GD α = 0,01)")
    ax[1].set_yscale("log"); ax[1].set_xlabel("warstwa liniowa"); ax[1].set_ylabel("std wyjścia warstwy")
    ax[1].set_title("Przejście w przód na starcie")
    ax[2].set_yscale("log"); ax[2].set_xlabel("warstwa liniowa"); ax[2].set_ylabel("||dL/dW||")
    ax[2].set_title("Gradient na starcie")
    for a in ax:
        a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "3_initialization.png"), dpi=130); plt.close()
    RESULTS["initialization"] = init_res
    print("[init]")
    for k, v in init_res.items():
        print(f"   {k:<20}", {kk: vv for kk, vv in v.items() if kk != "runs"})

    # ------------------------------------------------------------------ 6.4 Prosty gradient vs ADAM
    opt_res = {}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, (start, kind, sc) in enumerate([("He", "he", 1.0), ("N(0; 0,01)", "normal", 0.01)]):
        opt_res[start] = {}
        for label, make in OPTS:
            runs = [train(MLP(d, NET, kind, sc, seed=s), make(), Xtr, ytr, Xva, yva, mode="full", epochs=E_OPT, seed=s)
                    for s in SEEDS]
            sums = [summarize(r, D) for r in runs]
            opt_res[start][label] = dict(runs=sums, plateau_exit=[s["plateau_exit"] for s in sums],
                                         **{k: mean_of(sums, k) for k in ["epochs_to_target", "final_loss",
                                                                          "best_val_loss", "best_epoch", "test_f1_early"]})
            ax[i].plot(runs[0].history["epoch"], runs[0].history["loss"], label=label)
        ax[i].set_xscale("symlog"); ax[i].set_yscale("log"); ax[i].set_xlabel("epoka"); ax[i].set_ylabel("koszt treningowy")
        ax[i].set_title(f"Start: {start} (pełny zbiór)"); ax[i].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "4_gd_vs_adam.png"), dpi=130); plt.close()
    RESULTS["gd_vs_adam"] = opt_res
    print("[gd vs adam]")
    for start, v in opt_res.items():
        for k, r in v.items():
            print(f"   {start:<11} {k:<15}", {kk: vv for kk, vv in r.items() if kk != "runs"})

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

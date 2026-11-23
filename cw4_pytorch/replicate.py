# -*- coding: utf-8 -*-
"""
SIECI NEURONOWE - ćwiczenie 4, cz. 2: wszystkie eksperymenty z ćw. 3 (cz. 1 i cz. 2) powtórzone w PyTorch.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu należy go
 oznaczyć, zrozumieć i umieć samodzielnie wyjaśnić.]

Ustawienia jak w ćw. 3, a różnice wyeliminowane w equivalence.py:
  - float64,
  - wagi startowe losowane tym samym kodem co w ćw. 3 (NumPy, te same ziarna) i skopiowane do nn.Linear,
  - koszt raportowany tak jak bce() z ćw. 3 (obcięcie p do [1e-12, 1 - 1e-12]); uczenie BCEWithLogitsLoss,
  - paczki w tej samej kolejności (sampler z np.random.default_rng(seed).permutation).
Wyniki zapisywane w tej samej strukturze co results.json z ćw. 3, żeby compare.py mógł je zestawić.

Uruchomienie:  python replicate.py [sekcja ...]   sekcje: baseline width scaling depth he saddle modes init adam
               (bez argumentów - wszystkie; wyniki -> results_torch/<sekcja>.json, figures/rep_*.png)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cw3_mlp as ref          # noqa: E402  (tylko dane i losowanie wag startowych)
import cw3_optim as ref2       # noqa: E402
from torch_mlp import (TorchMLP, activation_stds, fit_full, loss_fn, summarize_full,  # noqa: E402
                       summarize_modes, to_t, train_modes)

FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "results_torch")
os.makedirs(FIG, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

E, E_DEPTH, SEEDS = ref.E, ref.E_DEPTH, ref.SEEDS          # 30 000, 150 000, (0, 1, 2)
NET = ref2.NET


def mean_of(items, fn):
    vals = [v for v in (fn(i) for i in items) if v is not None]
    return float(np.mean(vals)) if vals else None


def data(scaling="standard"):
    (Xtr, ytr), (Xva, yva), (Xte, yte), _ = ref.load_data(scaling)
    return [tuple(to_t(X, y)) for X, y in ((Xtr, ytr), (Xva, yva), (Xte, yte))]


def net1(d, hidden, seed, init="normal"):
    """Sieć z ćw. 3 cz. 1: wagi z MLP(...) z NumPy (N(0; 0,01) albo He) -> nn.Linear."""
    return TorchMLP.from_numpy(ref.MLP(d, hidden, init=init, seed=seed))


def net2(d, hidden, seed, init="he", scale=1.0):
    return TorchMLP.from_numpy(ref2.MLP(d, hidden, init, scale, seed=seed))


def save(name, obj):
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------- ćw. 3 cz. 1
def sec_baseline():
    T = data(); d = T[0][0].shape[1]
    m = fit_full(net1(d, (d,), 0), *T[0], *T[1], epochs=E)
    h = m.history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="walidacja")
    ax[0].axvline(m.best_epoch, c="tab:red", ls="--", label=f"min. kosztu walidacyjnego ({m.best_epoch})")
    ax[0].set_xscale("log"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("entropia krzyżowa"); ax[0].legend()
    ax[0].set_title("PyTorch: MLP 17-17-1, koszt")
    for k, ls in [("val_f1", "-"), ("val_precision", "--"), ("val_recall", ":")]:
        ax[1].plot(h["log_epoch"], h[k], ls, label=k)
    ax[1].set_xscale("log"); ax[1].set_ylim(0, 1); ax[1].set_xlabel("epoka"); ax[1].legend(); ax[1].set_title("Metryki (walidacja)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "rep_1_baseline.png"), dpi=130); plt.close()
    return summarize_full(m, T)


def sec_width():
    T = data(); d = T[0][0].shape[1]
    res, curves = {}, {}
    for w in [2, 4, d, 4 * d, 256]:
        runs = [fit_full(net1(d, (w,), s), *T[0], *T[1], epochs=E, log_every=100) for s in SEEDS]
        sums = [summarize_full(r, T) for r in runs]
        res[str(w)] = dict(runs=sums,
                           plateau_exit=mean_of(sums, lambda s: s["plateau_exit"]),
                           final_train_loss=mean_of(sums, lambda s: s["final_train_loss"]),
                           best_val_loss=mean_of(sums, lambda s: s["best_val_loss"]),
                           best_epoch=mean_of(sums, lambda s: s["best_epoch"]),
                           final_val_loss=mean_of(sums, lambda s: s["final_val_loss"]),
                           test_f1_early=mean_of(sums, lambda s: s["early_stopping"]["test"]["f1"]),
                           test_f1_final=mean_of(sums, lambda s: s["final"]["test"]["f1"]))
        curves[w] = (np.mean([r.history["loss"] for r in runs], 0), np.mean([r.history["val_loss"] for r in runs], 0))
        print("  width", w, {k: v for k, v in res[str(w)].items() if k != "runs"}, flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for w, (a, b) in curves.items():
        ax[0].plot(a, label=f"{w} neuronów"); ax[1].plot(b, label=f"{w} neuronów")
    for a, t in zip(ax, ["koszt treningowy", "koszt walidacyjny"]):
        a.set_xscale("log"); a.set_xlabel("epoka"); a.set_ylabel(t); a.legend(fontsize=8)
    ax[0].set_yscale("log"); ax[0].set_title("PyTorch: szerokość warstwy, train")
    ax[1].set_ylim(0.3, 1.5); ax[1].set_title("PyTorch: szerokość warstwy, walidacja")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "rep_2_width.png"), dpi=130); plt.close()
    return res


def sec_scaling():
    res = {}
    for sc, _ in ref.SCALINGS:
        T = data(sc); d = T[0][0].shape[1]
        res[sc] = {"cond_XtX": float(torch.linalg.cond(T[0][0].T @ T[0][0]))}
        for arch, hidden in [("logreg", ()), ("mlp", (d,))]:
            m = fit_full(net1(d, hidden, 0), *T[0], *T[1], epochs=E, log_every=100)
            res[sc][arch] = summarize_full(m, T)
        print("  scaling", sc, res[sc]["mlp"]["final_train_loss"], flush=True)
    return res


def sec_depth():
    T = data(); d = T[0][0].shape[1]
    res = {}
    for n in [0, 1, 2, 3, 4]:
        runs = [fit_full(net1(d, (d,) * n, s), *T[0], *T[1], epochs=E_DEPTH, log_every=1000) for s in SEEDS]
        sums = [summarize_full(r, T) for r in runs]
        res[str(n)] = dict(runs=sums, plateau_exit=[s["plateau_exit"] for s in sums],
                           final_train_loss=mean_of(sums, lambda s: s["final_train_loss"]),
                           best_val_loss=mean_of(sums, lambda s: s["best_val_loss"]),
                           test_f1_early=mean_of(sums, lambda s: s["early_stopping"]["test"]["f1"]),
                           grad_norms_start=runs[0].history["grad_norms"][0])
        print("  depth", n, {k: v for k, v in res[str(n)].items() if k != "runs"}, flush=True)
    return res


def sec_he():
    T = data(); d = T[0][0].shape[1]
    res = {}
    for n in [3, 4]:
        for init in ["normal", "he"]:
            m = fit_full(net1(d, (d,) * n, 0, init), *T[0], *T[1], epochs=E, log_every=100)
            res[f"{n}_{init}"] = summarize_full(m, T)
    return res


# ---------------------------------------------------------------------------- ćw. 3 cz. 2
def saddle_f(p):
    x, y = p[0], p[1]
    return x * x + y ** 4 / 4 - y * y / 2


def run_saddle(make_opt, start=(1.0, 1e-4), steps=150):
    """Ten sam optymalizator torcha, ale funkcja dwóch zmiennych zamiast sieci - gradient z autograd."""
    p = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    opt = make_opt([p])
    traj = [p.detach().clone()]
    for _ in range(steps):
        opt.zero_grad()
        saddle_f(p).backward()
        opt.step()
        traj.append(p.detach().clone())
    return torch.stack(traj).numpy()


def sec_saddle():
    trajs = {"GD (α = 0,1)": run_saddle(lambda ps: torch.optim.SGD(ps, lr=0.1)),
             "ADAM (α = 0,1)": run_saddle(lambda ps: torch.optim.Adam(ps, lr=0.1))}
    out = {}
    for k, t in trajs.items():
        t_np = ref2.run_saddle(ref2.GD(0.1) if k.startswith("GD") else ref2.Adam(0.1))
        out[k] = dict(escape_step=int(np.argmax(np.abs(t[:, 1]) > 0.5)), final=t[-1].tolist(),
                      final_f=float(saddle_f(t[-1])), traj_maxabs_vs_numpy=float(np.max(np.abs(t - t_np))))
    return out


def sec_modes():
    T = data(); d = T[0][0].shape[1]
    res = {}
    for mode in ["full", "minibatch", "single"]:
        runs = []
        for s in SEEDS:
            m = net2(d, NET, s)
            runs.append(train_modes(m, torch.optim.SGD(m.parameters(), lr=0.01), *T[0], *T[1], mode=mode,
                                    epochs=ref2.E_MODE, seed=s))
        sums = [summarize_modes(r, T) for r in runs]
        res[mode] = dict(runs=sums, steps_per_epoch=sums[0]["steps"] // ref2.E_MODE,
                         **{k: mean_of(sums, lambda x, k=k: x.get(k)) for k in
                            ["epochs_to_target", "steps_to_target", "time_to_target", "best_val_loss", "best_epoch",
                             "test_f1_early", "final_loss", "time_s"]})
        print("  mode", mode, {k: v for k, v in res[mode].items() if k != "runs"}, flush=True)
    return res


def sec_init():
    T = data(); d = T[0][0].shape[1]
    res = {}
    for label, kind, sc in ref2.INITS:
        models = [net2(d, NET, s, kind, sc) for s in SEEDS]
        stds = activation_stds(models[0], T[0][0])
        models[0].zero_grad()
        loss_fn(models[0](T[0][0]), T[0][1]).backward()
        grads0 = [float(l.weight.grad.norm()) for l in models[0].linears]
        runs = [train_modes(m, torch.optim.SGD(m.parameters(), lr=0.01), *T[0], *T[1], mode="minibatch",
                            epochs=ref2.E_INIT, seed=s) for m, s in zip(models, SEEDS)]
        sums = [summarize_modes(r, T) for r in runs]
        res[label] = dict(runs=sums, activation_std=stds, grad_norms_start=grads0,
                          n_diverged=int(sum(s["diverged"] for s in sums)),
                          **{k: mean_of(sums, lambda x, k=k: x.get(k)) for k in
                             ["start_loss", "final_loss", "plateau_exit", "epochs_to_target", "best_val_loss", "test_f1_early"]})
        print("  init", label, {k: v for k, v in res[label].items() if k != "runs"}, flush=True)
    return res


OPTS = [("GD α = 0,01", lambda p: torch.optim.SGD(p, lr=0.01)),
        ("ADAM α = 0,001", lambda p: torch.optim.Adam(p, lr=0.001)),
        ("ADAM α = 0,01", lambda p: torch.optim.Adam(p, lr=0.01))]


def sec_adam():
    T = data(); d = T[0][0].shape[1]
    res = {}
    for start, kind, sc in [("He", "he", 1.0), ("N(0; 0,01)", "normal", 0.01)]:
        res[start] = {}
        for label, make in OPTS:
            runs = []
            for s in SEEDS:
                m = net2(d, NET, s, kind, sc)
                runs.append(train_modes(m, make(m.parameters()), *T[0], *T[1], mode="full", epochs=ref2.E_OPT, seed=s))
            sums = [summarize_modes(r, T) for r in runs]
            res[start][label] = dict(runs=sums, plateau_exit=[s["plateau_exit"] for s in sums],
                                     **{k: mean_of(sums, lambda x, k=k: x.get(k)) for k in
                                        ["epochs_to_target", "final_loss", "best_val_loss", "best_epoch", "test_f1_early"]})
            print("  adam", start, label, {k: v for k, v in res[start][label].items() if k != "runs"}, flush=True)
    return res


SECTIONS = dict(baseline=sec_baseline, width=sec_width, scaling=sec_scaling, depth=sec_depth, he=sec_he,
                saddle=sec_saddle, modes=sec_modes, init=sec_init, adam=sec_adam)

if __name__ == "__main__":
    for name in sys.argv[1:] or SECTIONS:
        print(f"[{name}]", flush=True)
        save(name, SECTIONS[name]())

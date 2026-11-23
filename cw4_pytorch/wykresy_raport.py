# -*- coding: utf-8 -*-
"""
SIECI NEURONOWE - ćwiczenie 4: wykresy do krótkiego raportu (raport_cw4_prezentacja.pdf).

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu.]

Liczby w tabelach raportu pochodzą z results_*.json (equivalence.py, compare.py, diagnose.py, cifar10.py).
Krzywe, których nie zapisano w JSON, są tu liczone ponownie tym samym kodem (ok. 2 min na CPU).

Uruchomienie:  python wykresy_raport.py      (wyniki -> figures/raport_*.png)
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
import cw3_mlp as ref          # noqa: E402
import cw3_optim as ref2       # noqa: E402
from diagnose import nudge     # noqa: E402
from torch_mlp import TorchMLP, fit_full, to_t, train_modes  # noqa: E402

FIG = os.path.join(HERE, "figures")
E = 30_000
C = dict(blue="#4C72B0", orange="#DD8452", green="#55A868", red="#C44E52", gray="#8C8C8C", purple="#8172B3")

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "legend.frameon": True,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.edgecolor": "#bbbbbb",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.3,
})


def save(fig, name, title):
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(FIG, name), dpi=200)
    plt.close(fig)
    print("zapisano", name)


def load(name):
    return json.load(open(os.path.join(HERE, name), encoding="utf-8"))


def fig1_behaviour(T32, Xtr, ytr, Xva, yva, d):
    """K0 / K1 / ćw. 3: koszt treningowy i walidacyjny, ziarno 0."""
    runs = {}
    m = fit_full(TorchMLP(d, (d,), init="torch", seed=0, dtype=torch.float32), *T32[0], *T32[1], epochs=E)
    runs["K0: PyTorch, domyślne wagi, float32"] = (m.history["loss"], m.history["val_loss"], C["orange"], "-")
    m = fit_full(TorchMLP(d, (d,), init="normal", init_scale=0.01, seed=0, dtype=torch.float32),
                 *T32[0], *T32[1], epochs=E)
    runs["K1: PyTorch, N(0; 0,01), float32"] = (m.history["loss"], m.history["val_loss"], C["green"], "--")
    r = ref.MLP(d, (d,), seed=0).fit(Xtr, ytr, Xva, yva, epochs=E)
    runs["ćw. 3 (NumPy)"] = (r.history["loss"], r.history["val_loss"], C["blue"], "-")
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.7))
    for name in ["ćw. 3 (NumPy)", "K0: PyTorch, domyślne wagi, float32", "K1: PyTorch, N(0; 0,01), float32"]:
        tr, va, c, ls = runs[name]
        ax[0].plot(np.arange(1, len(tr) + 1), tr, color=c, ls=ls, label=name)
        ax[1].plot(np.arange(1, len(va) + 1), va, color=c, ls=ls, label=name)
    for a, t, yl in [(ax[0], "Koszt treningowy", "koszt treningowy"), (ax[1], "Koszt walidacyjny", "koszt walidacyjny")]:
        a.set_xscale("log"); a.set_xlabel("epoka (skala log.)"); a.set_ylabel(yl); a.set_title(t)
    ax[0].legend(loc="lower left")
    ax[1].set_ylim(0.3, 1.0)
    save(fig, "raport_1_naiwny_port.png", "Rys. 1. Naiwny port i ta sama inicjalizacja (sieć 17-17-1, ziarno 0)")


def fig2_levels(eq):
    """Maksymalna różnica |PyTorch − NumPy| na kolejnych poziomach, te same wagi startowe."""
    fw, gr, st, tr = eq["K2_K3_forward"], eq["K4_gradients"], eq["K6_optimizer_step"], eq["K7_training"]
    rows = [
        ("wagi startowe", fw["float32"]["weights_maxabs"], fw["float64"]["weights_maxabs"]),
        ("wyjście sieci (logity)", fw["float32"]["logits_maxabs"], fw["float64"]["logits_maxabs"]),
        ("koszt", fw["float32"]["loss_absdiff"], fw["float64"]["loss_absdiff"]),
        ("gradient dW (start)", gr["init N(0; 0,01), 1 warstwa | float32"]["dW_maxabs"],
         gr["init N(0; 0,01), 1 warstwa | float64"]["dW_maxabs"]),
        ("gradient dW (3 warstwy, He)", gr["3 warstwy ukryte, He | float32"]["dW_maxabs"],
         gr["3 warstwy ukryte, He | float64"]["dW_maxabs"]),
        ("wagi po 50 krokach SGD", None, st["SGD"]["after_50_steps"]),
        ("wagi po 50 krokach Adam", None, st["Adam"]["after_50_steps"]),
        ("koszt w całym uczeniu (30 000 ep.)", tr["float32"]["loss_maxabs"], tr["float64"]["loss_maxabs"]),
        ("wagi po całym uczeniu", tr["float32"]["final_weights_maxabs"], tr["float64"]["final_weights_maxabs"]),
    ]
    fig, ax = plt.subplots(figsize=(8, 2.9))
    y = np.arange(len(rows))[::-1]
    v32 = [r[1] if r[1] is not None else np.nan for r in rows]
    v64 = [max(r[2], 1e-19) for r in rows]
    ax.barh(y + 0.19, v32, height=0.36, color=C["orange"], label="float32 (domyślny w PyTorch)")
    ax.barh(y - 0.19, v64, height=0.36, color=C["blue"], label="float64 (jak NumPy)")
    for yi, a, b in zip(y, v32, rows):
        if b[2] == 0:
            ax.text(2e-19, yi - 0.19, "0 (identyczne)", va="center", fontsize=6.5, color="#333")
    ax.axvline(2.2e-16, color=C["gray"], ls=":", lw=1)
    ax.text(2.6e-16, len(rows) - 0.55, "1 ULP float64 (2,2·10⁻¹⁶)", fontsize=6.5, color="#555")
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xscale("log"); ax.set_xlim(1e-19, 1e-2); ax.set_ylim(-0.6, len(rows) - 0.2); ax.set_xlabel("maks. |PyTorch − NumPy| (skala log.)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    save(fig, "raport_2_poziomy.png", "Rys. 2. Te same wagi startowe: różnica na kolejnych poziomach (K2–K7)")


def fig3_training(T, Xtr, ytr, Xva, yva, d):
    """K7: różnica kosztu w czasie, float32 / float64; K8: kolejność przykładów w paczkach."""
    r = ref.MLP(d, (d,), seed=0).fit(Xtr, ytr, Xva, yva, epochs=E)
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.7))
    for name, dt, c in [("float32", torch.float32, C["orange"]), ("float64", torch.float64, C["blue"])]:
        TT = [tuple(to_t(X, y, dtype=dt)) for X, y in ((Xtr, ytr), (Xva, yva))]
        m = fit_full(TorchMLP.from_numpy(ref.MLP(d, (d,), seed=0), dtype=dt), *TT[0], *TT[1], epochs=E)
        diff = np.maximum(np.abs(np.array(m.history["loss"]) - np.array(r.history["loss"])), 1e-18)
        ax[0].plot(np.arange(1, E + 1), diff, color=c, lw=0.6, label=name)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(1e-18, 1e-4)
    ax[0].set_xlabel("epoka (skala log.)"); ax[0].set_ylabel("|koszt PyTorch − koszt NumPy|")
    ax[0].set_title("K7: całe uczenie, sieć 17-17-1"); ax[0].legend(loc="upper left")

    (Ytr, ttr), (Yva, tva), _, _ = ref2.load_data()
    T2 = [tuple(to_t(X, y)) for X, y in ((Ytr, ttr), (Yva, tva))]
    rm = ref2.train(ref2.MLP(d, ref2.NET, "he", seed=0), ref2.GD(0.01), Ytr, ttr, Yva, tva,
                    mode="minibatch", epochs=300, seed=0)
    for order, label, c in [("torch", "DataLoader(shuffle=True)", C["red"]),
                            ("numpy", "DataLoader + kolejność z NumPy", C["blue"])]:
        mt = TorchMLP.from_numpy(ref2.MLP(d, ref2.NET, "he", seed=0))
        train_modes(mt, torch.optim.SGD(mt.parameters(), lr=0.01), *T2[0], *T2[1], mode="minibatch",
                    epochs=300, seed=0, order=order)
        diff = np.maximum(np.abs(np.array(mt.history["loss"]) - np.array(rm.history["loss"])), 1e-18)
        ax[1].plot(np.array(mt.history["epoch"]) + 1, diff, color=c, lw=0.9, label=label)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_ylim(1e-18, 1e-1)
    ax[1].set_xlabel("epoka (skala log.)"); ax[1].set_ylabel("|koszt PyTorch − koszt NumPy|")
    ax[1].set_title("K8: paczki po 32, sieć 5 × 17, He"); ax[1].legend(loc="center right")
    save(fig, "raport_3_uczenie.png", "Rys. 3. Te same wagi startowe: różnica kosztu w trakcie uczenia")


def fig4_compare(cmp):
    names = {
        "cz. 1: model bazowy 17-17-1": "model bazowy 17-17-1",
        "cz. 1: szerokość warstwy (5 x 3 ziarna)": "szerokość warstwy",
        "cz. 1: skalowanie danych": "skalowanie danych",
        "cz. 1: liczba warstw (150 000 epok)": "liczba warstw",
        "cz. 1: inicjalizacja He": "inicjalizacja He",
        "cz. 2: punkt siodłowy": "punkt siodłowy",
        "cz. 2: full / paczki / pojedyncze": "pełny zbiór / paczki / pojedyncze",
        "cz. 2: inicjalizacja wag": "inicjalizacja wag",
        "cz. 2: GD vs ADAM": "GD vs ADAM",
    }
    rows = parse_compare_table()
    fig, ax = plt.subplots(figsize=(8, 2.6))
    y = np.arange(len(rows))[::-1]
    for yi, (n, tot, same, close, diff, _) in zip(y, rows):
        a, b, c = same / tot, close / tot, diff / tot
        ax.barh(yi, a, color=C["blue"], label="identyczne co do bitu" if yi == y[0] else None)
        ax.barh(yi, b, left=a, color="#9fb7d9", label="różnica ≤ 10⁻⁹ (w praktyce ~10⁻¹⁶)" if yi == y[0] else None)
        ax.barh(yi, c, left=a + b, color=C["red"], label="różne" if yi == y[0] else None)
        ax.text(1.01, yi, f"{tot}", va="center", fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([names.get(r[0], r[0]) for r in rows])
    ax.set_xlim(0, 1.06); ax.set_xlabel("udział porównanych liczb (z prawej: ile liczb porównano)")
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_xticklabels([f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.5, frameon=False)
    save(fig, "raport_4_eksperymenty.png", "Rys. 4. Wszystkie liczby z ćw. 3 porównane z PyTorch (2598 liczb)")


def parse_compare_table():
    rows = []
    for line in open(os.path.join(HERE, "compare_table.md"), encoding="utf-8"):
        p = [s.strip() for s in line.strip().strip("|").split("|")]
        if len(p) == 6 and p[1].isdigit():
            rows.append((p[0], int(p[1]), int(p[2]), int(p[3]), int(p[4]), p[5]))
    return rows


def fig5_sensitivity(d):
    (Ra, ra), (Rb, rb), _, _ = ref.load_data("none")
    TR = [tuple(to_t(Ra, ra)), tuple(to_t(Rb, rb))]
    a = ref.MLP(d, (), seed=0).fit(Ra, ra, Rb, rb, epochs=ref.E, log_every=1000)
    b = nudge(ref.MLP(d, (), seed=0)).fit(Ra, ra, Rb, rb, epochs=ref.E, log_every=1000)
    t = fit_full(TorchMLP.from_numpy(ref.MLP(d, (), seed=0)), *TR[0], *TR[1], epochs=ref.E, log_every=1000)
    la, lb, lt = (np.array(m.history["loss"]) for m in (a, b, t))
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.7))
    ep = np.arange(1, len(la) + 1)
    ax[0].plot(ep, np.maximum(np.abs(la - lt), 1e-18), color=C["blue"], lw=0.5, label="NumPy vs PyTorch")
    ax[0].plot(ep, np.maximum(np.abs(la - lb), 1e-18), color=C["orange"], lw=0.5, alpha=0.75,
               label="NumPy vs NumPy (wagi +1 ULP)")
    ax[0].set_title("Dane surowe, bez warstwy ukrytej")

    (Xtr, ytr), (Xva, yva), _, _ = ref2.load_data()
    T = [tuple(to_t(X, y)) for X, y in ((Xtr, ytr), (Xva, yva))]
    mk = lambda: ref2.MLP(d, ref2.NET, "normal", 0.01, seed=2)     # noqa: E731
    a = ref2.train(mk(), ref2.Adam(0.001), Xtr, ytr, Xva, yva, mode="full", epochs=ref2.E_OPT, seed=2)
    b = ref2.train(nudge(mk()), ref2.Adam(0.001), Xtr, ytr, Xva, yva, mode="full", epochs=ref2.E_OPT, seed=2)
    mt = TorchMLP.from_numpy(mk())
    train_modes(mt, torch.optim.Adam(mt.parameters(), lr=0.001), *T[0], *T[1], mode="full", epochs=ref2.E_OPT, seed=2)
    la, lb, lt = (np.array(m.history["loss"]) for m in (a, b, mt))
    ep = np.arange(1, len(la) + 1)
    ax[1].plot(ep, np.maximum(np.abs(la - lt), 1e-18), color=C["blue"], lw=0.7, label="NumPy vs PyTorch")
    ax[1].plot(ep, np.maximum(np.abs(la - lb), 1e-18), color=C["orange"], lw=0.7, alpha=0.75,
               label="NumPy vs NumPy (wagi +1 ULP)")
    ax[1].set_title("ADAM α = 0,001, start N(0; 0,01), ziarno 2")
    for a_ in ax:
        a_.set_xscale("log"); a_.set_yscale("log"); a_.set_xlabel("epoka (skala log.)")
        a_.set_ylabel("|różnica kosztu|"); a_.legend(loc="upper left")
    save(fig, "raport_5_czulosc.png", "Rys. 5. Przebiegi wrażliwe numerycznie: PyTorch a zmiana wag o 1 ULP")


def fig6_cifar(cf):
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.7))
    for (name, r), c in zip(cf["runs"].items(), [C["red"], C["blue"]]):
        h = r["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax[0].plot(ep, h["train_loss"], color=c, label=f"{name}: trening")
        ax[0].plot(ep, h["val_loss"], color=c, ls="--", label=f"{name}: walidacja")
        ax[1].plot(ep, h["train_acc"], color=c, label=f"{name}: trening")
        ax[1].plot(ep, h["val_acc"], color=c, ls="--", label=f"{name}: walidacja")
        ax[1].axvline(r["best_epoch"], color=c, ls=":", lw=0.9)
    ax[0].set_title("Koszt (entropia krzyżowa)"); ax[0].set_ylabel("koszt")
    ax[1].set_title("Dokładność (linia pionowa: wybrana epoka)"); ax[1].set_ylabel("dokładność")
    for a in ax:
        a.set_xlabel("epoka")
    ax[0].legend(loc="upper left"); ax[1].legend(loc="lower right")
    save(fig, "raport_6_cifar.png", "Rys. 6. CIFAR10: MLP 3072-1024-512-256-10, ADAM α = 0,001, paczki po 128")

    pl = {"airplane": "samolot", "automobile": "samochód", "bird": "ptak", "cat": "kot", "deer": "jeleń",
          "dog": "pies", "frog": "żaba", "horse": "koń", "ship": "statek", "truck": "ciężarówka"}
    r = cf["runs"]["MLP + dropout 0,3"]
    cm = np.array(r["confusion"], dtype=float)
    cm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw=dict(width_ratios=[1, 1.35]))
    names = list(r["per_class_acc"].keys())
    acc = np.array([r["per_class_acc"][k] for k in names])
    o = np.argsort(acc)
    ax[0].barh(np.arange(10), acc[o], color=C["blue"])
    ax[0].set_yticks(np.arange(10)); ax[0].set_yticklabels([pl[names[i]] for i in o])
    ax[0].axvline(r["test_acc"], color=C["red"], ls="--", lw=1)
    ax[0].text(r["test_acc"] + 0.01, 0, f"średnio {r['test_acc']:.3f}".replace(".", ","), fontsize=6.5, color=C["red"])
    ax[0].set_xlim(0, 0.8); ax[0].set_xlabel("dokładność na teście"); ax[0].set_title("Dokładność per klasa")
    ax[0].grid(axis="y", visible=False)
    im = ax[1].imshow(cm, cmap="Blues", vmin=0, vmax=0.8)
    for i in range(10):
        for j in range(10):
            ax[1].text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.5,
                       color="white" if cm[i, j] > 0.45 else "#333")
    lab = [pl[k] for k in names]
    ax[1].set_xticks(range(10)); ax[1].set_xticklabels(lab, rotation=45, ha="right", fontsize=6)
    ax[1].set_yticks(range(10)); ax[1].set_yticklabels(lab, fontsize=6)
    ax[1].set_xlabel("predykcja"); ax[1].set_ylabel("prawdziwa klasa"); ax[1].grid(False)
    ax[1].set_title("Macierz pomyłek (% wiersza)")
    save(fig, "raport_7_cifar_klasy.png", "Rys. 7. CIFAR10, MLP + dropout 0,3: zbiór testowy (10 000 obrazków)")


def main():
    which = set(sys.argv[1:]) or {"1", "2", "3", "4", "5", "6"}
    (Xtr, ytr), (Xva, yva), (Xte, yte), _ = ref.load_data("standard")
    d = Xtr.shape[1]
    T32 = [tuple(to_t(X, y, dtype=torch.float32)) for X, y in ((Xtr, ytr), (Xva, yva))]
    if "2" in which:
        fig2_levels(load("results_equivalence.json"))
    if "4" in which:
        fig4_compare(None)
    if "6" in which:
        fig6_cifar(load("results_cifar10.json"))
    if "1" in which:
        fig1_behaviour(T32, Xtr, ytr, Xva, yva, d)
    if "3" in which:
        fig3_training(None, Xtr, ytr, Xva, yva, d)
    if "5" in which:
        fig5_sensitivity(d)


if __name__ == "__main__":
    main()

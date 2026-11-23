# -*- coding: utf-8 -*-
"""
SIECI NEURONOWE - ćwiczenie 4: wyjaśnienie różnic, które zostały po replikacji (compare.py).

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu.]

D1  "Zepsuta" inicjalizacja N(0; 10): w NumPy przebieg kończy się NaN, w PyTorch nie zawsze.
    Hipoteza: ReLU z ćw. 3 liczy X * [X > 0], a dla X = -inf daje -inf * 0 = NaN; nn.ReLU daje 0.
    Test: (a) torch z ReLU zapisanym jak w ćw. 3 (MaskReLU), (b) NumPy z ReLU = np.maximum(X, 0).
D2  Dane surowe, regresja logistyczna: wynik po 30 000 epok inny o kilka procent.
    Hipoteza: uczenie jest niestabilne (alpha za duże dla nieskalowanych cech), więc różnice rzędu 1e-16
    rosną wykładniczo. Test: NumPy kontra NumPy z wagami startowymi zmienionymi o 1 ULP (najmniejszy krok float64).
D3  ADAM, start N(0; 0,01): różnice ~1e-8 w koszcie końcowym. Ten sam test z ULP.

Uruchomienie:  python diagnose.py     (wyniki -> results_diagnose.json, figures/diag_*.png)
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
from torch_mlp import TorchMLP, fit_full, summarize_modes, to_t, train_modes  # noqa: E402

FIG = os.path.join(HERE, "figures")
R = {}
KEYS = ["diverged", "epochs_run", "final_loss", "best_val_loss", "best_epoch", "test_f1_early"]


def nudge(np_model):
    """Każda waga przesunięta o 1 ULP w górę - różnica ~1e-18 przy wagach ~0,01."""
    for l in np_model.linears:
        l.W = np.nextafter(l.W, np.inf)
    return np_model


def main():
    (Xtr, ytr), (Xva, yva), (Xte, yte), _ = ref2.load_data()
    d = Xtr.shape[1]
    D = ((Xtr, ytr), (Xva, yva), (Xte, yte))
    T = [tuple(to_t(X, y)) for X, y in D]

    # ------------------------------------------------------------------ D1
    out = {}
    with np.errstate(all="ignore"):
        for s in ref2.SEEDS:
            row = {}
            m = ref2.train(ref2.MLP(d, ref2.NET, "normal", 10.0, seed=s), ref2.GD(0.01), Xtr, ytr, Xva, yva,
                           mode="minibatch", epochs=ref2.E_INIT, seed=s)
            row["numpy (X * maska)"] = ref2.summarize(m, D)
            orig = ref2.ReLU.forward
            ref2.ReLU.forward = lambda self, X: (setattr(self, "mask", X > 0), np.maximum(X, 0.0))[1]
            m = ref2.train(ref2.MLP(d, ref2.NET, "normal", 10.0, seed=s), ref2.GD(0.01), Xtr, ytr, Xva, yva,
                           mode="minibatch", epochs=ref2.E_INIT, seed=s)
            ref2.ReLU.forward = orig
            row["numpy (np.maximum)"] = ref2.summarize(m, D)
            for relu, name in [("torch", "torch (nn.ReLU)"), ("mask", "torch (MaskReLU)")]:
                mt = TorchMLP.from_numpy(ref2.MLP(d, ref2.NET, "normal", 10.0, seed=s), relu=relu)
                train_modes(mt, torch.optim.SGD(mt.parameters(), lr=0.01), *T[0], *T[1], mode="minibatch",
                            epochs=ref2.E_INIT, seed=s)
                row[name] = summarize_modes(mt, T)
            out[f"seed {s}"] = {k: {kk: v[kk] for kk in KEYS} for k, v in row.items()}
            print(f"[D1 seed {s}]")
            for k, v in out[f"seed {s}"].items():
                print(f"   {k:<20}", v)
    R["D1_broken_init_relu"] = out
    R["D1_ieee"] = dict(inf_times_zero=float(np.float64(-np.inf) * 0.0),
                        torch_relu_minus_inf=float(torch.relu(torch.tensor(-np.inf))),
                        mask_relu_minus_inf=float(torch.tensor(-np.inf) * (torch.tensor(-np.inf) > 0)))
    print("[D1 IEEE]", R["D1_ieee"])

    # ------------------------------------------------------------------ D2 / D3: czułość na 1 ULP
    (Ra, ra), (Rb, rb), _, _ = ref.load_data("none")
    TR = [tuple(to_t(Ra, ra)), tuple(to_t(Rb, rb))]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    a = ref.MLP(d, (), seed=0).fit(Ra, ra, Rb, rb, epochs=ref.E, log_every=1000)
    b = nudge(ref.MLP(d, (), seed=0)).fit(Ra, ra, Rb, rb, epochs=ref.E, log_every=1000)
    t = fit_full(TorchMLP.from_numpy(ref.MLP(d, (), seed=0)), *TR[0], *TR[1], epochs=ref.E, log_every=1000)
    la, lb, lt = (np.array(m.history["loss"]) for m in (a, b, t))
    R["D2_raw_logreg"] = dict(final_loss_numpy=la[-1], final_loss_numpy_1ulp=lb[-1], final_loss_torch=lt[-1],
                              loss_curve_std_last_1000=float(la[-1000:].std()),
                              first_epoch_diff_numpy_vs_torch_gt_1e6=int(np.argmax(np.abs(la - lt) > 1e-6)),
                              first_epoch_diff_numpy_vs_1ulp_gt_1e6=int(np.argmax(np.abs(la - lb) > 1e-6)))
    print("[D2]", R["D2_raw_logreg"])
    ax[0].plot(np.maximum(np.abs(la - lt), 1e-18), label="NumPy vs PyTorch", lw=0.8)
    ax[0].plot(np.maximum(np.abs(la - lb), 1e-18), label="NumPy vs NumPy (+1 ULP na wagach)", lw=0.8, alpha=0.7)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("|różnica kosztu|")
    ax[0].set_title("D2: dane surowe, bez warstwy ukrytej, α = 0,01"); ax[0].legend(fontsize=8)

    d3 = {}
    for s in ref2.SEEDS:
        mk = lambda: ref2.MLP(d, ref2.NET, "normal", 0.01, seed=s)     # noqa: E731
        a = ref2.train(mk(), ref2.Adam(0.001), Xtr, ytr, Xva, yva, mode="full", epochs=ref2.E_OPT, seed=s)
        b = ref2.train(nudge(mk()), ref2.Adam(0.001), Xtr, ytr, Xva, yva, mode="full", epochs=ref2.E_OPT, seed=s)
        mt = TorchMLP.from_numpy(mk())
        train_modes(mt, torch.optim.Adam(mt.parameters(), lr=0.001), *T[0], *T[1], mode="full", epochs=ref2.E_OPT, seed=s)
        la, lb, lt = (np.array(m.history["loss"]) for m in (a, b, mt))
        d3[f"seed {s}"] = dict(final_numpy=la[-1], final_numpy_1ulp=lb[-1], final_torch=lt[-1],
                               maxdiff_numpy_torch=float(np.abs(la - lt).max()), maxdiff_numpy_1ulp=float(np.abs(la - lb).max()))
        if s == 2:
            ax[1].plot(np.maximum(np.abs(la - lt), 1e-18), label="NumPy vs PyTorch")
            ax[1].plot(np.maximum(np.abs(la - lb), 1e-18), label="NumPy vs NumPy (+1 ULP)", alpha=0.7)
    ax[1].set_xscale("symlog"); ax[1].set_yscale("log"); ax[1].set_xlabel("epoka"); ax[1].set_ylabel("|różnica kosztu|")
    ax[1].set_title("D3: ADAM α = 0,001, start N(0; 0,01), 5 warstw, ziarno 2"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "diag_sensitivity.png"), dpi=130); plt.close()
    R["D3_adam_small_init"] = d3
    print("[D3]", d3)

    with open(os.path.join(HERE, "results_diagnose.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()

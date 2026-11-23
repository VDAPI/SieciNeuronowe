# -*- coding: utf-8 -*-
"""
SIECI NEURONOWE - ćwiczenie 4, cz. 1: krok po kroku od "naiwnego" przepisania sieci z ćw. 3 na PyTorch
do implementacji dającej (prawie) bit w bit te same liczby.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu należy go
 oznaczyć, zrozumieć i umieć samodzielnie wyjaśnić.]

Każdy krok usuwa jedną różnicę implementacyjną i sprawdza podobieństwo na kolejnym poziomie:
  K0  naiwny port (jak w tutorialu): domyślna inicjalizacja nn.Linear, float32     -> zachowanie modelu
  K1  ta sama inicjalizacja co w ćw. 3 (N(0; 0,01), bias 0), ale generator torcha -> zachowanie / metryki
  K2  te same wagi (skopiowane z NumPy), float32                                    -> wyjście sieci, koszt
  K3  float64 zamiast float32                                                        -> wyjście, koszt
  K4  gradienty: autograd vs ręczny backward z ćw. 3                                -> gradienty
  K5  funkcja kosztu: BCEWithLogitsLoss vs obcinane bce() z ćw. 3                   -> koszt przy skrajnych logitach
  K6  jeden krok optymalizatora: torch.optim.SGD / Adam vs własne GD / Adam          -> wagi po kroku
  K7  całe uczenie (30 000 epok, pełny zbiór) float64 i float32                      -> krzywa kosztu, wagi, metryki
  K8  uczenie paczkami: kolejność przykładów (DataLoader shuffle vs np.permutation)  -> krzywa kosztu, wagi

Uruchomienie:  python equivalence.py      (wyniki -> results_equivalence.json, figures/eq_*.png)
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cw3_mlp as ref          # noqa: E402  kod z ćw. 3 cz. 1 (bez zmian)
import cw3_optim as ref2       # noqa: E402  kod z ćw. 3 cz. 2 (bez zmian)
from torch_mlp import (TorchMLP, bce_report, fit_full, loss_fn, metrics, summarize_full,  # noqa: E402
                       to_t, train_modes)

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
R = {}
E = 30_000


def maxabs(a, b):
    a = a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
    b = b.detach().cpu().numpy() if torch.is_tensor(b) else np.asarray(b)
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def weights_diff(np_model, t_model):
    return max(max(maxabs(ln.W, lt.weight), maxabs(ln.b, lt.bias))
               for ln, lt in zip(np_model.linears, t_model.linears))


def short(s):
    return {k: s[k] for k in ["final_train_loss", "best_val_loss", "best_epoch", "plateau_exit"]} | \
        {"test_f1_early": s["early_stopping"]["test"]["f1"], "test_f1_final": s["final"]["test"]["f1"]}


def main():
    (Xtr, ytr), (Xva, yva), (Xte, yte), _ = ref.load_data("standard")
    d = Xtr.shape[1]
    D_np = ((Xtr, ytr), (Xva, yva), (Xte, yte))
    T64 = [tuple(to_t(X, y)) for X, y in D_np]
    T32 = [tuple(to_t(X, y, dtype=torch.float32)) for X, y in D_np]

    # ------------------------------------------------------------------ punkt odniesienia: ćw. 3 (NumPy)
    t0 = time.perf_counter()
    ref_m = ref.MLP(d, (d,), seed=0).fit(Xtr, ytr, Xva, yva, epochs=E)
    t_np = time.perf_counter() - t0
    ref_s = ref.summarize(ref_m, D_np)
    stored = json.load(open(os.path.join(HERE, "dane", "cw3_results.json"), encoding="utf-8"))["baseline"]
    R["reference"] = dict(numpy=short(ref_s), stored_cw3=short(stored), time_s=t_np)
    print("[ref] numpy:", R["reference"]["numpy"])
    print("      zapis z ćw. 3:", R["reference"]["stored_cw3"])

    # ------------------------------------------------------------------ K0 naiwny port
    seeds = [0, 1, 2]
    k0, k1 = [], []
    curves = {}
    for s in seeds:
        m = fit_full(TorchMLP(d, (d,), init="torch", seed=s, dtype=torch.float32), *T32[0], *T32[1], epochs=E)
        k0.append(short(summarize_full(m, T32)))
        if s == 0:
            curves["K0: domyślna inicjalizacja torcha, float32"] = m.history["loss"]
        m = fit_full(TorchMLP(d, (d,), init="normal", init_scale=0.01, seed=s, dtype=torch.float32),
                     *T32[0], *T32[1], epochs=E)
        k1.append(short(summarize_full(m, T32)))
        if s == 0:
            curves["K1: N(0; 0,01) generatorem torcha, float32"] = m.history["loss"]
    np_runs = [short(ref.summarize(ref.MLP(d, (d,), seed=s).fit(Xtr, ytr, Xva, yva, epochs=E), D_np)) for s in seeds]
    curves["ćw. 3 (NumPy)"] = ref_m.history["loss"]
    R["K0_naive"], R["K1_same_init_distribution"], R["numpy_seeds"] = k0, k1, np_runs
    lin0 = torch.nn.Linear(d, d)
    R["K0_default_init_std"] = dict(weight_std=float(lin0.weight.detach().std()), bias_std=float(lin0.bias.detach().std()),
                                    expected_uniform_bound=1 / np.sqrt(d))
    for name, runs in [("numpy", np_runs), ("K0", k0), ("K1", k1)]:
        print(f"[{name}]", [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in runs])

    # ------------------------------------------------------------------ K2 / K3 te same wagi: przejście w przód
    m_np = ref.MLP(d, (d,), seed=0)
    fw = {}
    for name, dt, T in [("float32", torch.float32, T32), ("float64", torch.float64, T64)]:
        mt = TorchMLP.from_numpy(m_np, dtype=dt)
        z_np = m_np.forward(Xtr)
        z_t = mt(T[0][0])
        L_np = m_np.loss_fn.forward(z_np, ytr)
        fw[name] = dict(weights_maxabs=weights_diff(m_np, mt), logits_maxabs=maxabs(z_np, z_t),
                        logits_rel=maxabs(z_np, z_t) / float(np.max(np.abs(z_np))),
                        loss_np=L_np, loss_torch=float(loss_fn(z_t, T[0][1])), loss_absdiff=abs(L_np - float(loss_fn(z_t, T[0][1]))))
        # wytrenowana sieć: większe wagi -> większe logity, łatwiej zobaczyć różnice
        mt2 = TorchMLP.from_numpy(ref_m, dtype=dt)
        fw[name]["trained_logits_maxabs"] = maxabs(ref_m.forward(Xte), mt2(T[2][0]))
        fw[name]["trained_same_predictions"] = bool(np.array_equal(ref_m.predict_proba(Xte) >= 0.5,
                                                                   (mt2.predict_proba(T[2][0]) >= 0.5).numpy()))
    R["K2_K3_forward"] = fw
    print("[K2/K3 forward]", fw)

    # ------------------------------------------------------------------ K4 gradienty autograd vs ręczny backward
    gr = {}
    for label, model_np in [("init N(0; 0,01), 1 warstwa", ref.MLP(d, (d,), seed=0)),
                            ("wytrenowana (30k epok)", ref_m),
                            ("3 warstwy ukryte, He", ref.MLP(d, (d,) * 3, init="he", seed=1))]:
        for name, dt, T in [("float32", torch.float32, T32), ("float64", torch.float64, T64)]:
            mt = TorchMLP.from_numpy(model_np, dtype=dt)
            model_np.loss_and_grad(Xtr, ytr)
            mt.zero_grad()
            z = mt(T[0][0])
            z.retain_grad()
            loss_fn(z, T[0][1]).backward()
            dz_np = model_np.loss_fn.backward()[:, 0]
            gW = max(maxabs(ln.dW, lt.weight.grad) for ln, lt in zip(model_np.linears, mt.linears))
            gb = max(maxabs(ln.db, lt.bias.grad) for ln, lt in zip(model_np.linears, mt.linears))
            gmax = max(float(np.max(np.abs(ln.dW))) for ln in model_np.linears)
            gr[f"{label} | {name}"] = dict(dL_dz_maxabs=maxabs(dz_np, z.grad), dW_maxabs=gW, db_maxabs=gb,
                                           dW_rel=gW / gmax, grad_fn=type(z.grad_fn).__name__)
    R["K4_gradients"] = gr
    for k, v in gr.items():
        print(f"[K4] {k:<42} dW={v['dW_maxabs']:.2e} (wzgl. {v['dW_rel']:.1e}) db={v['db_maxabs']:.2e}")

    # ------------------------------------------------------------------ K5 funkcja kosztu przy skrajnych logitach
    z = np.array([-50.0, -30.0, -10.0, 0.0, 10.0, 30.0, 50.0])
    yb = np.ones_like(z)
    sm = ref.SigmoidBCE()
    loss_rows = []
    for zi, yi in zip(z, yb):
        L_np = sm.forward(np.array([zi]), np.array([yi]))
        zt, yt = to_t([zi], [yi])
        zt.requires_grad_(True)
        L_bce_logits = loss_fn(zt, yt)
        L_bce_logits.backward()
        g_logits = float(zt.grad[0])
        zt2 = torch.tensor([zi], dtype=torch.float64, requires_grad=True)
        L_bce = torch.nn.BCELoss()(torch.sigmoid(zt2), yt)        # sigmoida + BCELoss osobno
        L_bce.backward()
        loss_rows.append(dict(z=zi, y=yi, numpy_bce_clip=L_np, torch_bce_report_clip=bce_report(zt.detach(), yt, "clip"),
                              BCEWithLogitsLoss=float(L_bce_logits), sigmoid_BCELoss=float(L_bce),
                              grad_numpy=float(sm.backward()[0, 0]), grad_BCEWithLogits=g_logits,
                              grad_sigmoid_BCELoss=float(zt2.grad[0])))
    R["K5_loss_extremes"] = loss_rows
    print("[K5] z, numpy, BCEWithLogits, sigmoid+BCELoss, grad numpy / BCEWithLogits / sigmoid+BCELoss")
    for r in loss_rows:
        print(f"     z={r['z']:>6} L: {r['numpy_bce_clip']:8.3f} {r['BCEWithLogitsLoss']:8.3f} {r['sigmoid_BCELoss']:8.3f}"
              f"   grad: {r['grad_numpy']: .3e} {r['grad_BCEWithLogits']: .3e} {r['grad_sigmoid_BCELoss']: .3e}")
    # wpływ na raportowany koszt walidacyjny przeuczonej sieci (256 neuronów, koniec uczenia)
    big = ref.MLP(d, (256,), seed=0).fit(Xtr, ytr, Xva, yva, epochs=E, log_every=1000)
    mt = TorchMLP.from_numpy(big)
    zv = mt(T64[1][0]).detach()
    R["K5_overfit_256_val_loss"] = dict(numpy=ref.bce(yva, big.predict_proba(Xva)), torch_clip=bce_report(zv, T64[1][1], "clip"),
                                        torch_native=bce_report(zv, T64[1][1], "native"),
                                        max_abs_logit=float(zv.abs().max()))
    print("[K5] koszt walidacyjny sieci 256 po 30k epokach:", R["K5_overfit_256_val_loss"])

    # ------------------------------------------------------------------ K6 jeden krok optymalizatora
    st = {}
    for oname in ["SGD", "Adam"]:
        m_np = ref2.MLP(d, (d,) * 2, "he", seed=3)
        mt = TorchMLP.from_numpy(m_np)
        opt_np = ref2.GD(0.01) if oname == "SGD" else ref2.Adam(0.01)
        opt_t = torch.optim.SGD(mt.parameters(), lr=0.01) if oname == "SGD" else torch.optim.Adam(mt.parameters(), lr=0.01)
        diffs = []
        for k in range(50):
            idx = np.arange(k * 8, k * 8 + 32) % len(ytr)
            m_np.loss_and_grad(Xtr[idx], ytr[idx]); opt_np.step(m_np.params_and_grads())
            opt_t.zero_grad(); loss_fn(mt(T64[0][0][idx]), T64[0][1][idx]).backward(); opt_t.step()
            diffs.append(weights_diff(m_np, mt))
        st[oname] = dict(after_1_step=diffs[0], after_10_steps=diffs[9], after_50_steps=diffs[-1])
    R["K6_optimizer_step"] = st
    print("[K6]", st)

    # ------------------------------------------------------------------ K7 całe uczenie, te same wagi
    traj = {}
    for name, dt, T in [("float64", torch.float64, T64), ("float32", torch.float32, T32)]:
        t0 = time.perf_counter()
        mt = fit_full(TorchMLP.from_numpy(ref.MLP(d, (d,), seed=0), dtype=dt), *T[0], *T[1], epochs=E)
        tt = time.perf_counter() - t0
        s = summarize_full(mt, T)
        ld = np.abs(np.array(mt.history["loss"]) - np.array(ref_m.history["loss"]))
        traj[name] = dict(summary=short(s), time_s=tt, loss_maxabs=float(ld.max()),
                          loss_maxabs_first_1000=float(ld[:1000].max()),
                          val_loss_maxabs=maxabs(mt.history["val_loss"], ref_m.history["val_loss"]),
                          final_weights_maxabs=weights_diff(ref_m, mt),
                          best_weights_same_epoch=bool(mt.best_epoch == ref_m.best_epoch),
                          test_confusion_same=all(s[w]["test"][k] == ref_s[w]["test"][k]
                                                  for w in ["final", "early_stopping"] for k in ["tp", "fp", "fn", "tn"]))
        traj[name]["_loss_diff_curve"] = ld
        print(f"[K7 {name}]", {k: v for k, v in traj[name].items() if not k.startswith("_")})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, c in curves.items():
        ax[0].plot(c, label=name)
    ax[0].set_xscale("log"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("koszt treningowy (seed 0)")
    ax[0].set_title("Zachowanie modelu: K0 / K1 vs ćw. 3"); ax[0].legend(fontsize=8)
    for name in ["float64", "float32"]:
        ax[1].plot(np.maximum(traj[name]["_loss_diff_curve"], 1e-18), label=f"te same wagi startowe, {name}")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("epoka")
    ax[1].set_ylabel("|koszt torch − koszt NumPy|"); ax[1].set_title("K7: różnica kosztu w trakcie uczenia")
    ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "eq_1_full_batch.png"), dpi=130); plt.close()
    for v in traj.values():
        v.pop("_loss_diff_curve")
    R["K7_training"] = traj

    # ------------------------------------------------------------------ K8 paczki: kolejność przykładów
    mb = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    ref_mb = ref2.train(ref2.MLP(d, ref2.NET, "he", seed=0), ref2.GD(0.01), Xtr, ytr, Xva, yva,
                        mode="minibatch", epochs=300, seed=0)
    ax.plot(ref_mb.history["epoch"], ref_mb.history["loss"], lw=3, alpha=0.4, label="ćw. 3 cz. 2 (NumPy)")
    for order, label in [("torch", "DataLoader(shuffle=True)"), ("numpy", "DataLoader + sampler z np.permutation")]:
        mt = TorchMLP.from_numpy(ref2.MLP(d, ref2.NET, "he", seed=0))
        train_modes(mt, torch.optim.SGD(mt.parameters(), lr=0.01), *T64[0], *T64[1], mode="minibatch",
                    epochs=300, seed=0, order=order)
        mb[order] = dict(loss_maxabs=maxabs(mt.history["loss"], ref_mb.history["loss"]),
                         final_weights_maxabs=weights_diff(ref_mb, mt), final_loss=mt.history["loss"][-1],
                         final_loss_numpy=ref_mb.history["loss"][-1])
        ax.plot(mt.history["epoch"], mt.history["loss"], "--", label=label)
    ax.set_xscale("symlog"); ax.set_yscale("log"); ax.set_xlabel("epoka"); ax.set_ylabel("koszt treningowy")
    ax.set_title("K8: paczki po 32, sieć 17-(17x5)-1, He, te same wagi startowe"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "eq_2_minibatch_order.png"), dpi=130); plt.close()
    R["K8_minibatch_order"] = mb
    print("[K8]", mb)

    with open(os.path.join(HERE, "results_equivalence.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()

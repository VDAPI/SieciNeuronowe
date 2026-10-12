# -*- coding: utf-8 -*-
"""
SIECI NEURONOWE - ćwiczenie 2: regresja logistyczna na zbiorze Heart Disease.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu należy go
 oznaczyć, zrozumieć i umieć samodzielnie wyjaśnić.]

Model:   p(x) = sigma(W x + b)
Koszt:   L = -1/N * sum[ y ln p + (1-y) ln(1-p) ]      (średnia po przykładach)
Gradient: dL/dw_i = -1/N * sum (y - p) x_i,   dL/db = -1/N * sum (y - p)
Aktualizacja: w <- w - alpha * dL/dw            (pełny batch gradient descent)

Uruchomienie:  python logreg.py      (wyniki -> figures/ oraz results.json)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 42
RESULTS = {}


# ----------------------------------------------------------------------------
# 1. Dane
# ----------------------------------------------------------------------------
import sys
sys.path.insert(0, HERE)
if hasattr(sys.stdout, "reconfigure"):  # Windows: avoid UnicodeEncodeError (cp1250) when output is redirected
    sys.stdout.reconfigure(encoding="utf-8")
from data import load_splits  # noqa: E402  (preprocessing taki sam jak w ćwiczeniu 1)


def load_data(standardize=True, clean=True, seed=SEED):
    """clean=False -> braki kodowane dodatkowymi kolumnami-wskaźnikami (ca_brak, thal_brak) i bez łączenia
    rzadkiej kategorii restecg=1. Kolumna thal_brak ma w zbiorze treningowym jedną jedynkę -> patrz eksperyment 3.7."""
    return load_splits(seed=seed, standardize=standardize, merge_rare_restecg=clean, missing_indicators=not clean)


# ----------------------------------------------------------------------------
# 2. Model
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


def init_params(d, kind="zeros", scale=0.01, rng=None):
    rng = rng or np.random.default_rng(SEED)
    if kind == "zeros":
        w = np.zeros(d)
    elif kind == "normal":
        w = rng.normal(0.0, scale, d)
    elif kind == "uniform":
        w = rng.uniform(-scale, scale, d)
    elif kind == "const":
        w = np.full(d, scale)
    else:
        raise ValueError(kind)
    return w, 0.0


class LogisticRegressionGD:
    def __init__(self, lr=0.1, epochs=1000, init="zeros", init_scale=0.01,
                 stop=None, tol=1e-6, patience=50, seed=SEED):
        """stop: None -> stała liczba epok
                 'tol'  -> stop gdy |L_t - L_{t-1}| < tol (koszt treningowy)
                 'early'-> early stopping: brak poprawy kosztu walidacyjnego przez `patience` epok"""
        self.lr, self.epochs, self.init, self.init_scale = lr, epochs, init, init_scale
        self.stop, self.tol, self.patience = stop, tol, patience
        self.rng = np.random.default_rng(seed)

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def fit(self, X, y, Xv=None, yv=None):
        self.w, self.b = init_params(X.shape[1], self.init, self.init_scale, self.rng)
        N = len(y)
        h = {k: [] for k in ["loss", "val_loss", "f1", "val_f1", "precision", "recall",
                             "val_precision", "val_recall", "grad_norm"]}
        best = (np.inf, None, None, 0)
        self.stopped_at = self.epochs
        for ep in range(self.epochs):
            p = self.predict_proba(X)
            err = y - p
            gw = -(X.T @ err) / N          # dL/dw
            gb = -np.mean(err)             # dL/db
            # --- logowanie (przed krokiem: stan modelu w epoce ep) ---
            h["loss"].append(bce(y, p))
            m = metrics(y, p)
            h["f1"].append(m["f1"]); h["precision"].append(m["precision"]); h["recall"].append(m["recall"])
            h["grad_norm"].append(float(np.sqrt(gw @ gw + gb * gb)))
            if Xv is not None:
                pv = self.predict_proba(Xv)
                h["val_loss"].append(bce(yv, pv))
                mv = metrics(yv, pv)
                h["val_f1"].append(mv["f1"]); h["val_precision"].append(mv["precision"]); h["val_recall"].append(mv["recall"])
            # --- warunki stopu ---
            if self.stop == "tol" and ep > 0 and abs(h["loss"][-2] - h["loss"][-1]) < self.tol:
                self.stopped_at = ep; break
            if self.stop == "early":
                if h["val_loss"][-1] < best[0] - 1e-9:
                    best = (h["val_loss"][-1], self.w.copy(), self.b, ep)
                elif ep - best[3] >= self.patience:
                    self.w, self.b = best[1], best[2]   # przywrócenie najlepszych wag
                    self.stopped_at = ep; self.best_epoch = best[3]; break
            # --- krok gradientowy ---
            self.w -= self.lr * gw
            self.b -= self.lr * gb
        self.history = h
        return self


def epochs_to(loss_hist, target):
    """Pierwsza epoka, w której koszt spadł poniżej target (None jeśli nigdy)."""
    idx = np.where(np.asarray(loss_hist) <= target)[0]
    return int(idx[0]) if len(idx) else None


# ----------------------------------------------------------------------------
# 3. Eksperymenty
# ----------------------------------------------------------------------------
def main():
    (Xtr, ytr), (Xva, yva), (Xte, yte), names = load_data()
    print(f"train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}; cech: {len(names)}")
    print(f"udział klasy 1: train {ytr.mean():.2f}, val {yva.mean():.2f}, test {yte.mean():.2f}")
    RESULTS["data"] = dict(n_train=len(ytr), n_val=len(yva), n_test=len(yte),
                           n_features=len(names), pos_rate_train=float(ytr.mean()))

    # --- Punkt odniesienia: minimum globalne (bardzo długi trening, mały/stabilny krok).
    # Funkcja kosztu jest wypukła -> to minimum jest jedyne (o ile dane nie są liniowo separowalne).
    ref = LogisticRegressionGD(lr=0.5, epochs=200_000, stop="tol", tol=1e-14).fit(Xtr, ytr)
    L_star = ref.history["loss"][-1]
    print(f"[ref] L* = {L_star:.6f} po {ref.stopped_at} epokach, |grad| = {ref.history['grad_norm'][-1]:.2e}")
    RESULTS["reference"] = dict(L_star=L_star, epochs=ref.stopped_at,
                                grad_norm=ref.history["grad_norm"][-1],
                                test=metrics(yte, ref.predict_proba(Xte)))
    target = L_star + 1e-3   # "zbieżność" = koszt w odległości 1e-3 od minimum

    # ------------------------------------------------------------------ 3.1 Baseline
    base = LogisticRegressionGD(lr=0.1, epochs=2000, init="zeros").fit(Xtr, ytr, Xva, yva)
    h = base.history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="walidacja")
    ax[0].axhline(L_star, ls="--", c="gray", lw=1, label="L* (minimum, train)")
    ax[0].set_xlabel("epoka"); ax[0].set_ylabel("entropia krzyżowa"); ax[0].set_title("Krzywa uczenia - koszt")
    ax[0].legend(); ax[0].set_xscale("log")
    for k, ls in [("f1", "-"), ("precision", "--"), ("recall", ":")]:
        ax[1].plot(h["val_" + k], ls, label=f"{k} (walidacja)")
    ax[1].plot(h["f1"], c="k", alpha=.4, label="f1 (train)")
    ax[1].set_xlabel("epoka"); ax[1].set_title("Krzywa uczenia - metryki"); ax[1].set_xscale("log")
    ax[1].set_ylim(0.4, 1.0); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "1_baseline_learning_curve.png"), dpi=130); plt.close()
    RESULTS["baseline"] = dict(lr=0.1, epochs=2000, init="zeros",
                               train=metrics(ytr, base.predict_proba(Xtr)),
                               val=metrics(yva, base.predict_proba(Xva)),
                               test=metrics(yte, base.predict_proba(Xte)),
                               final_loss=h["loss"][-1], epochs_to_converge=epochs_to(h["loss"], target))
    print("[baseline] test:", RESULTS["baseline"]["test"])

    # wagi modelu (interpretacja)
    order = np.argsort(np.abs(base.w))[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([names[i] for i in order][::-1], base.w[order][::-1],
            color=["tab:red" if v < 0 else "tab:blue" for v in base.w[order][::-1]])
    ax.set_title("Wagi modelu (cechy standaryzowane)"); ax.axvline(0, c="k", lw=.8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "2_weights.png"), dpi=130); plt.close()

    # ------------------------------------------------------------------ 3.2 Współczynnik uczenia
    lrs = [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 20.0, 100.0]
    E = 5000
    lr_res = {}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for lr in lrs:
        m = LogisticRegressionGD(lr=lr, epochs=E).fit(Xtr, ytr, Xva, yva)
        hh = m.history
        lr_res[lr] = dict(final_loss=hh["loss"][-1], epochs_to_converge=epochs_to(hh["loss"], target),
                          min_loss=min(hh["loss"]),
                          oscillating=bool(np.sum(np.diff(hh["loss"][-200:]) > 0) > 20),
                          val_f1=metrics(yva, m.predict_proba(Xva))["f1"],
                          test_f1=metrics(yte, m.predict_proba(Xte))["f1"])
        ax[0].plot(hh["loss"], label=f"α={lr:g}")
        ax[1].plot(hh["val_f1"], label=f"α={lr:g}")
    ax[0].axhline(L_star, ls="--", c="gray", lw=1)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("koszt (train)")
    ax[0].set_title("Wpływ współczynnika uczenia na koszt"); ax[0].legend(fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("epoka"); ax[1].set_ylabel("F1 (walidacja)")
    ax[1].set_title("Wpływ współczynnika uczenia na F1"); ax[1].set_ylim(0.4, 1.0)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "3_learning_rate.png"), dpi=130); plt.close()
    RESULTS["learning_rate"] = {str(k): v for k, v in lr_res.items()}
    print("[lr]"); [print(f"   α={k:<8g} {v}") for k, v in lr_res.items()]

    # zbliżenie na pierwsze epoki dla dużych α (oscylacje)
    fig, ax = plt.subplots(figsize=(7, 4))
    for lr in [1.0, 5.0, 20.0, 100.0]:
        m = LogisticRegressionGD(lr=lr, epochs=60).fit(Xtr, ytr)
        ax.plot(m.history["loss"], marker=".", label=f"α={lr:g}")
    ax.axhline(L_star, ls="--", c="gray", lw=1); ax.set_yscale("log")
    ax.set_xlabel("epoka"); ax.set_ylabel("koszt (train)"); ax.set_title("Duże α - pierwsze 60 epok"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "4_large_lr_zoom.png"), dpi=130); plt.close()

    # ------------------------------------------------------------------ 3.3 Inicjalizacja parametrów
    inits = [("zeros", 0.0), ("normal", 0.01), ("normal", 1.0), ("normal", 10.0),
             ("uniform", 1.0), ("uniform", 10.0), ("const", 5.0)]
    n_seeds = 10
    init_res = {}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for kind, sc in inits:
        label = "zeros" if kind == "zeros" else f"{kind}({sc:g})"
        runs = []
        for s in range(n_seeds if kind in ("normal", "uniform") else 1):
            m = LogisticRegressionGD(lr=0.1, epochs=5000, init=kind, init_scale=sc, seed=s).fit(Xtr, ytr, Xva, yva)
            runs.append(m)
        L = np.array([r.history["loss"] for r in runs])
        conv = [epochs_to(r.history["loss"], target) for r in runs]
        init_res[label] = dict(
            init_loss_mean=float(L[:, 0].mean()), final_loss_mean=float(L[:, -1].mean()),
            final_loss_std=float(L[:, -1].std()),
            epochs_to_converge_mean=float(np.mean([c for c in conv if c is not None])) if any(c is not None for c in conv) else None,
            n_not_converged=int(sum(c is None for c in conv)),
            val_f1_after_100=float(np.mean([r.history["val_f1"][100] for r in runs])),
            test_f1_final=float(np.mean([metrics(yte, r.predict_proba(Xte))["f1"] for r in runs])),
            w_dist_to_ref=float(np.mean([np.linalg.norm(r.w - ref.w) for r in runs])))
        line, = ax[0].plot(L.mean(0), label=label)
        if len(runs) > 1:
            ax[0].fill_between(range(L.shape[1]), L.min(0), L.max(0), color=line.get_color(), alpha=.15)
        ax[1].plot(np.mean([r.history["val_f1"] for r in runs], 0), label=label)
    ax[0].axhline(L_star, ls="--", c="gray", lw=1)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("koszt (train)")
    ax[0].set_title(f"Inicjalizacja (α=0.1; średnia i zakres z {n_seeds} ziaren)"); ax[0].legend(fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("epoka"); ax[1].set_ylabel("F1 (walidacja)"); ax[1].set_ylim(0.3, 1.0)
    ax[1].set_title("Inicjalizacja - F1 walidacyjne"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "5_initialization.png"), dpi=130); plt.close()
    RESULTS["initialization"] = init_res
    print("[init]"); [print(f"   {k:<14} {v}") for k, v in init_res.items()]

    # ------------------------------------------------------------------ 3.4 Liczba epok / warunek stopu
    stop_res = {}
    for name, kw in [("stałe 10 epok", dict(epochs=10)), ("stałe 100 epok", dict(epochs=100)),
                     ("stałe 1000 epok", dict(epochs=1000)), ("stałe 20000 epok", dict(epochs=20000)),
                     ("tol 1e-4", dict(epochs=20000, stop="tol", tol=1e-4)),
                     ("tol 1e-6", dict(epochs=20000, stop="tol", tol=1e-6)),
                     ("early stopping (patience 50)", dict(epochs=20000, stop="early", patience=50))]:
        m = LogisticRegressionGD(lr=0.1, **kw).fit(Xtr, ytr, Xva, yva)
        stop_res[name] = dict(epochs_run=int(m.stopped_at), train_loss=bce(ytr, m.predict_proba(Xtr)),
                              val_loss=bce(yva, m.predict_proba(Xva)),
                              val=metrics(yva, m.predict_proba(Xva)), test=metrics(yte, m.predict_proba(Xte)))
        if kw.get("stop") == "early":
            stop_res[name]["best_epoch"] = int(m.best_epoch)
            es = m
    RESULTS["stopping"] = stop_res
    print("[stop]"); [print(f"   {k:<30} ep={v['epochs_run']:<6} trainL={v['train_loss']:.4f} valL={v['val_loss']:.4f} "
                            f"testF1={v['test']['f1']:.3f}") for k, v in stop_res.items()]

    long = LogisticRegressionGD(lr=0.1, epochs=20000).fit(Xtr, ytr, Xva, yva).history
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(long["loss"], label="train"); ax[0].plot(long["val_loss"], label="walidacja")
    ax[0].axvline(es.best_epoch, c="tab:red", ls="--", label=f"early stopping (najlepsza epoka {es.best_epoch})")
    ax[0].axvline(stop_res["tol 1e-4"]["epochs_run"], c="tab:green", ls=":", label="stop tol=1e-4")
    ax[0].axvline(stop_res["tol 1e-6"]["epochs_run"], c="tab:purple", ls=":", label="stop tol=1e-6")
    ax[0].set_xscale("log"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("koszt"); ax[0].legend(fontsize=8)
    ax[0].set_title("Warunki stopu na krzywej uczenia (α=0.1)")
    ax[1].plot(long["grad_norm"]); ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("epoka"); ax[1].set_ylabel("||∇L||"); ax[1].set_title("Norma gradientu")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "6_stopping.png"), dpi=130); plt.close()

    # ------------------------------------------------------------------ 3.5 Zła parametryzacja: brak standaryzacji
    (Xtr_r, ytr_r), (Xva_r, yva_r), (Xte_r, yte_r), _ = load_data(standardize=False)
    raw_res = {}
    fig, ax = plt.subplots(figsize=(7, 4))
    for lr in [1e-5, 1e-4, 1e-3, 1e-2]:
        m = LogisticRegressionGD(lr=lr, epochs=20000).fit(Xtr_r, ytr_r, Xva_r, yva_r)
        raw_res[str(lr)] = dict(final_loss=m.history["loss"][-1], min_loss=min(m.history["loss"]),
                                test_f1=metrics(yte_r, m.predict_proba(Xte_r))["f1"])
        ax.plot(m.history["loss"], label=f"surowe dane, α={lr:g}")
    ax.plot(long["loss"], c="k", lw=2, label="standaryzowane, α=0.1")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.2, 50)
    ax.set_xlabel("epoka"); ax.set_ylabel("koszt (train)"); ax.set_title("Wpływ braku standaryzacji cech"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "7_no_standardization.png"), dpi=130); plt.close()
    # minimum dla surowych danych jest TO SAMO co dla standaryzowanych (przekształcenie afiniczne),
    # różni się tylko uwarunkowanie problemu
    RESULTS["no_standardization"] = raw_res
    RESULTS["condition_number"] = dict(standardized=float(np.linalg.cond(Xtr.T @ Xtr)),
                                       raw=float(np.linalg.cond(Xtr_r.T @ Xtr_r)))
    print("[raw]", raw_res, RESULTS["condition_number"])

    # ------------------------------------------------------------------ 3.6 Kontrola: sklearn (jeśli jest)
    try:
        from sklearn.linear_model import LogisticRegression
        sk = LogisticRegression(C=np.inf, max_iter=100000, tol=1e-12).fit(Xtr, ytr)
        RESULTS["sklearn_check"] = dict(loss=bce(ytr, sk.predict_proba(Xtr)[:, 1]),
                                        w_diff=float(np.linalg.norm(sk.coef_[0] - ref.w)),
                                        test=metrics(yte, sk.predict_proba(Xte)[:, 1]))
        print("[sklearn]", RESULTS["sklearn_check"])
    except ImportError:
        pass

    # ------------------------------------------------------------------ 3.7 Kolumny-wskaźniki braków: minimum "w nieskończoności"
    (Xtr_u, ytr_u), _, (Xte_u, yte_u), names_u = load_data(clean=False)
    mu_ = LogisticRegressionGD(lr=0.5, epochs=200_000, stop="tol", tol=1e-14)
    # śledzimy normę wag i kilka wag "rzadkich" kategorii w trakcie uczenia
    ckpts = np.unique(np.logspace(0, np.log10(200_000), 60).astype(int))
    mu_.w, mu_.b = init_params(Xtr_u.shape[1]); traj, lossu = [], []
    for ep in range(1, 200_001):
        p = mu_.predict_proba(Xtr_u); err = ytr_u - p
        mu_.w += 0.5 * (Xtr_u.T @ err) / len(ytr_u); mu_.b += 0.5 * np.mean(err)
        if ep in ckpts:
            traj.append(mu_.w.copy()); lossu.append(bce(ytr_u, mu_.predict_proba(Xtr_u)))
    traj = np.array(traj)
    # to samo dla imputacji dominantą (bez wskaźników)
    mc = LogisticRegressionGD(lr=0.5, epochs=1).fit(Xtr, ytr)
    mc.w, mc.b = init_params(Xtr.shape[1]); trajc, lossc = [], []
    for ep in range(1, 200_001):
        p = mc.predict_proba(Xtr); err = ytr - p
        mc.w += 0.5 * (Xtr.T @ err) / len(ytr); mc.b += 0.5 * np.mean(err)
        if ep in ckpts:
            trajc.append(mc.w.copy()); lossc.append(bce(ytr, mc.predict_proba(Xtr)))
    trajc = np.array(trajc)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(ckpts, lossu, label="ze wskaźnikami braków"); ax[0].plot(ckpts, lossc, label="imputacja dominantą")
    ax[0].set_xscale("log"); ax[0].set_xlabel("epoka"); ax[0].set_ylabel("koszt (train)"); ax[0].legend()
    ax[0].set_title("Koszt (α=0.5)"); ax[0].set_ylim(0.2, 0.7)
    ax[1].plot(ckpts, np.linalg.norm(traj, axis=1), label="ze wskaźnikami braków")
    ax[1].plot(ckpts, np.linalg.norm(trajc, axis=1), label="imputacja dominantą")
    ax[1].set_xscale("log"); ax[1].set_xlabel("epoka"); ax[1].set_ylabel("||w||"); ax[1].legend()
    ax[1].set_title("Norma wektora wag")
    for nm in ["thal_brak", "ca_brak", "restecg_1", "thal_7", "cp_4"]:
        if nm in names_u:
            ax[2].plot(ckpts, traj[:, names_u.index(nm)], label=nm)
    ax[2].set_xscale("log"); ax[2].set_xlabel("epoka"); ax[2].set_ylabel("waga"); ax[2].legend(fontsize=8)
    ax[2].set_title("Wagi wybranych cech (ze wskaźnikami braków)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "8_rare_categories_divergence.png"), dpi=130); plt.close()
    RESULTS["unclean_data"] = dict(
        loss_at=dict(zip(["1e3", "1e4", "1e5", "2e5"], [float(np.interp(e, ckpts, lossu)) for e in (1e3, 1e4, 1e5, 2e5)])),
        wnorm_at=dict(zip(["1e3", "1e4", "1e5", "2e5"], [float(np.interp(e, ckpts, np.linalg.norm(traj, axis=1))) for e in (1e3, 1e4, 1e5, 2e5)])),
        clean_wnorm_at=dict(zip(["1e3", "1e4", "1e5", "2e5"], [float(np.interp(e, ckpts, np.linalg.norm(trajc, axis=1))) for e in (1e3, 1e4, 1e5, 2e5)])),
        test_f1=metrics(yte_u, mu_.predict_proba(Xte_u))["f1"])
    print("[unclean]", RESULTS["unclean_data"])

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

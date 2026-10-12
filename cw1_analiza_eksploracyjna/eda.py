# -*- coding: utf-8 -*-
"""
SIECI NEURONOWE - ćwiczenie 1: analiza eksploracyjna zbioru Heart Disease + baseline.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu należy go
 oznaczyć, zrozumieć i umieć samodzielnie wyjaśnić.]

Uruchomienie:  python eda.py   (wykresy -> figures/, liczby -> results.json)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if hasattr(sys.stdout, "reconfigure"):  # Windows: avoid UnicodeEncodeError (cp1250) when output is redirected
    sys.stdout.reconfigure(encoding="utf-8")
from data import (BINARY, CATEGORICAL, NUMERIC, TARGET, Preprocessor,  # noqa: E402
                  load_raw, split_indices)

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
R = {}


def main():
    df_all = load_raw(drop_duplicates=False)
    R["n_rows_raw"] = len(df_all)
    R["n_duplicates"] = int(df_all.duplicated().sum())
    df = load_raw()                       # bez duplikatów
    R["n_rows"] = len(df)
    print(f"Wiersze: {len(df_all)}, duplikaty: {R['n_duplicates']} -> {len(df)} po usunięciu")

    # ---------------------------------------------------------------- 1. Zbalansowanie klas
    counts = df[TARGET].value_counts().sort_index()
    R["class_counts"] = {int(k): int(v) for k, v in counts.items()}
    R["class_ratio"] = float(counts.min() / counts.max())
    print("Klasy:", R["class_counts"], f"stosunek mniejszej do większej = {R['class_ratio']:.2f}")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["0: brak choroby", "1: choroba"], counts.values, color=["tab:green", "tab:red"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_title("Liczność klas"); ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "1_klasy.png"), dpi=130); plt.close()

    # ---------------------------------------------------------------- 2. Średnie, odchylenia, normalność
    num = df[NUMERIC]
    desc = num.describe().T[["mean", "std", "min", "50%", "max"]]
    desc["skośność"] = num.skew()
    desc["kurtoza"] = num.kurt()
    sw = {c: stats.shapiro(num[c].dropna()) for c in NUMERIC}
    desc["Shapiro W"] = [sw[c].statistic for c in NUMERIC]
    desc["Shapiro p"] = [sw[c].pvalue for c in NUMERIC]
    desc["normalny (p>0.05)?"] = desc["Shapiro p"] > 0.05
    print("\nCechy liczbowe:\n", desc.round(3).to_string())
    R["numeric"] = desc.round(4).astype(object).to_dict(orient="index")
    R["numeric_by_class"] = df.groupby(TARGET)[NUMERIC].mean().round(2).to_dict()

    fig, axes = plt.subplots(2, len(NUMERIC), figsize=(3.2 * len(NUMERIC), 6))
    for i, c in enumerate(NUMERIC):
        x = num[c].dropna()
        axes[0, i].hist(x, bins=20, density=True, alpha=.6, color="tab:blue")
        g = np.linspace(x.min(), x.max(), 200)
        axes[0, i].plot(g, stats.norm.pdf(g, x.mean(), x.std()), "r", lw=1.5, label="N(μ,σ)")
        axes[0, i].set_title(f"{c}\nskośność={x.skew():.2f}, p={sw[c].pvalue:.1e}", fontsize=9)
        stats.probplot(x, dist="norm", plot=axes[1, i])
        axes[1, i].set_title(f"Q-Q {c}", fontsize=9); axes[1, i].set_xlabel(""); axes[1, i].set_ylabel("")
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "2_rozklady_liczbowe.png"), dpi=130); plt.close()

    fig, axes = plt.subplots(1, len(NUMERIC), figsize=(3.2 * len(NUMERIC), 3.3))
    for i, c in enumerate(NUMERIC):
        axes[i].boxplot([df.loc[df[TARGET] == k, c].dropna() for k in (0, 1)], tick_labels=["0", "1"])
        axes[i].set_title(c); axes[i].set_xlabel("klasa")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "3_liczbowe_vs_klasa.png"), dpi=130); plt.close()

    # ---------------------------------------------------------------- 3. Cechy kategoryczne: równomierność
    cat_res = {}
    cats = BINARY + CATEGORICAL
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, c in zip(axes.ravel(), cats):
        vc = df[c].value_counts().sort_index()
        chi = stats.chisquare(vc.values)          # H0: rozkład równomierny
        cat_res[c] = dict(counts={int(k): int(v) for k, v in vc.items()},
                          chi2=float(chi.statistic), p=float(chi.pvalue),
                          min_share=float(vc.min() / vc.sum()),
                          disease_rate={int(k): round(float(v), 3) for k, v in df.groupby(c)[TARGET].mean().items()})
        ct = pd.crosstab(df[c], df[TARGET])
        ct.plot.bar(stacked=True, ax=ax, color=["tab:green", "tab:red"], legend=False, rot=0)
        ax.axhline(len(df) / len(vc), c="k", ls="--", lw=1)
        ax.set_title(f"{c}  (χ² p={chi.pvalue:.1e})", fontsize=10); ax.set_xlabel("")
    axes[0, 0].legend(["klasa 0", "klasa 1", ], fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "4_rozklady_kategoryczne.png"), dpi=130); plt.close()
    R["categorical"] = cat_res
    print("\nCechy kategoryczne (test χ² zgodności z rozkładem równomiernym):")
    for c, v in cat_res.items():
        print(f"  {c:<8} {v['counts']}  p={v['p']:.2e}  odsetek chorych={v['disease_rate']}")

    # ---------------------------------------------------------------- 4. Braki danych
    miss = df.isna().sum()
    R["missing"] = {k: int(v) for k, v in miss[miss > 0].items()}
    R["rows_with_missing"] = int(df.isna().any(axis=1).sum())
    print("\nBraki:", R["missing"], "wierszy z brakiem:", R["rows_with_missing"])
    print(df[df.isna().any(axis=1)].to_string())

    # korelacje (po zakodowaniu)
    pre = Preprocessor().fit(df)
    Xall = pre.transform(df)
    corr = np.corrcoef(np.column_stack([Xall, df[TARGET].to_numpy(float)]).T)
    names = pre.feature_names_ + ["target"]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=8); ax.set_yticks(range(len(names)), names, fontsize=8)
    plt.colorbar(im, fraction=.046); ax.set_title("Korelacja Pearsona (po kodowaniu one-hot)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "5_korelacje.png"), dpi=130); plt.close()
    R["corr_with_target"] = dict(sorted({n: round(float(v), 3) for n, v in zip(names[:-1], corr[-1, :-1])}.items(),
                                        key=lambda kv: -abs(kv[1])))

    # ---------------------------------------------------------------- 5. Baseline w scikit-learn
    y = df[TARGET].to_numpy()
    tr, va, te = split_indices(y)
    trv = np.sort(np.r_[tr, va])          # baseline uczymy na train+val, oceniamy na test
    pre = Preprocessor().fit(df.iloc[trv])
    Xtr, Xte = pre.transform(df.iloc[trv]), pre.transform(df.iloc[te])
    models = {"Dummy (zawsze klasa większościowa)": DummyClassifier(strategy="most_frequent"),
              "LogisticRegression (sklearn)": LogisticRegression(max_iter=5000),
              "RandomForest (sklearn)": RandomForestClassifier(n_estimators=300, random_state=0)}
    base = {}
    for name, m in models.items():
        m.fit(Xtr, y[trv]); pr = m.predict(Xte); pp = m.predict_proba(Xte)[:, 1]
        # 5-krotna walidacja krzyżowa na całości (preprocessing dopasowywany w każdym foldzie)
        cv = []
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(df, y):
            p2 = Preprocessor().fit(df.iloc[a])
            m.fit(p2.transform(df.iloc[a]), y[a])
            cv.append(f1_score(y[b], m.predict(p2.transform(df.iloc[b])), zero_division=0))
        m.fit(Xtr, y[trv])
        base[name] = dict(accuracy=accuracy_score(y[te], pr), precision=precision_score(y[te], pr, zero_division=0),
                          recall=recall_score(y[te], pr), f1=f1_score(y[te], pr, zero_division=0),
                          roc_auc=roc_auc_score(y[te], pp), cv_f1_mean=float(np.mean(cv)), cv_f1_std=float(np.std(cv)),
                          confusion=confusion_matrix(y[te], pr).tolist())
    R["split"] = dict(train=len(tr), val=len(va), test=len(te))
    R["baseline"] = base
    print("\nBaseline (test, n=%d):" % len(te))
    for k, v in base.items():
        print(f"  {k:<36} acc={v['accuracy']:.3f} P={v['precision']:.3f} R={v['recall']:.3f} "
              f"F1={v['f1']:.3f} AUC={v['roc_auc']:.3f} | CV F1={v['cv_f1_mean']:.3f}±{v['cv_f1_std']:.3f}")

    with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()

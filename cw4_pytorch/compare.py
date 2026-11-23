# -*- coding: utf-8 -*-
"""
SIECI NEURONOWE - ćwiczenie 4: zestawienie wszystkich liczb z ćw. 3 (results.json, NumPy) z wynikami
replicate.py (PyTorch). Porównuje każdą liczbę (liść JSON) o tej samej ścieżce.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu.]

Uruchomienie:  python compare.py     (po replicate.py; wyniki -> results_compare.json, compare_table.md)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAIRS = [  # (plik ćw. 3, klucz w ćw. 3, sekcja replicate.py, opis)
    ("cw3_results.json", "baseline", "baseline", "cz. 1: model bazowy 17-17-1"),
    ("cw3_results.json", "hidden_width", "width", "cz. 1: szerokość warstwy (5 x 3 ziarna)"),
    ("cw3_results.json", "scaling", "scaling", "cz. 1: skalowanie danych"),
    ("cw3_results.json", "depth", "depth", "cz. 1: liczba warstw (150 000 epok)"),
    ("cw3_results.json", "he_init", "he", "cz. 1: inicjalizacja He"),
    ("cw3_cz2_results.json", "saddle", "saddle", "cz. 2: punkt siodłowy"),
    ("cw3_cz2_results.json", "batch_modes", "modes", "cz. 2: full / paczki / pojedyncze"),
    ("cw3_cz2_results.json", "initialization", "init", "cz. 2: inicjalizacja wag"),
    ("cw3_cz2_results.json", "gd_vs_adam", "adam", "cz. 2: GD vs ADAM"),
]
SKIP = ("time_s", "time_to_target")        # czas zależy od maszyny i biblioteki - nie porównujemy


def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def classify(a, b):
    """'same' (identyczne), 'close' (|różnica| <= 1e-9 * max(1, |a|)), 'diff', 'nan' (oba NaN = zgodne)."""
    if a is None or b is None or isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return ("same" if a == b else "diff"), (0.0 if a == b else np.inf)
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return "same", 0.0
    d = abs(a - b)
    if d == 0:
        return "same", 0.0
    return ("close" if d <= 1e-9 * max(1.0, abs(a)) else "diff"), d


def main():
    files = {f: json.load(open(os.path.join(HERE, "dane", f), encoding="utf-8")) for f in {p[0] for p in PAIRS}}
    out, rows = {}, []
    for fname, key, sec, desc in PAIRS:
        path = os.path.join(HERE, "results_torch", f"{sec}.json")
        if not os.path.exists(path):
            rows.append(f"| {desc} | brak wyników | | | | |")
            continue
        ref = dict(leaves(files[fname][key]))
        tor = dict(leaves(json.load(open(path, encoding="utf-8"))))
        stats = {"same": 0, "close": 0, "diff": 0}
        worst, diffs = 0.0, []
        for p, a in ref.items():
            if any(s in p for s in SKIP) or p not in tor:
                continue
            c, d = classify(a, tor[p])
            stats[c] += 1
            if c != "same" and np.isfinite(d):
                worst = max(worst, d)
            if c == "diff":
                diffs.append(dict(path=p, numpy=a, torch=tor[p]))
        n = sum(stats.values())
        out[sec] = dict(description=desc, compared=n, **stats, max_abs_diff=worst, differences=diffs)
        rows.append(f"| {desc} | {n} | {stats['same']} | {stats['close']} | {stats['diff']} | {worst:.1e} |")
    table = ("| eksperyment | porównanych liczb | identyczne | różnica ≤ 1e-9 | różne | maks. różnica |\n"
             "|---|---|---|---|---|---|\n" + "\n".join(rows))
    print(table)
    for sec, v in out.items():
        for d in v["differences"][:10]:
            print(f"  {sec}: {d['path']}: numpy={d['numpy']} torch={d['torch']}")
    with open(os.path.join(HERE, "results_compare.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(HERE, "compare_table.md"), "w", encoding="utf-8") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()

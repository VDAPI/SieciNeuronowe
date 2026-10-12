# -*- coding: utf-8 -*-
"""
Wczytanie i przygotowanie zbioru Heart Disease (Cleveland, UCI) - wspólne dla ćwiczeń 1-3.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu.]

Źródło: UCI Heart Disease, plik Cleveland (303 próbki, 14 atrybutów).
Strona UCI (archive.ics.uci.edu) nie była osiągalna ze środowiska, w którym przygotowano
kod, więc użyto kopii ARFF tego samego zbioru (heart-c.arff); konwersja do kodów
liczbowych UCI jest w heart_cleveland.csv (puste pola = braki, w oryginale '?').
target = 0 brak choroby (num == 0), 1 choroba (num in 1..4).

Opis kodów (zgodny z dokumentacją UCI):
  sex: 1 mężczyzna, 0 kobieta
  cp: 1 typowa dławica, 2 nietypowa dławica, 3 ból niedławicowy, 4 bezobjawowo
  fbs: cukier na czczo > 120 mg/dl (1/0)
  restecg: 0 norma, 1 nieprawidłowość ST-T, 2 przerost lewej komory
  exang: dławica wysiłkowa (1/0)
  slope: nachylenie ST: 1 wznoszące, 2 płaskie, 3 opadające
  ca: liczba naczyń zabarwionych we fluoroskopii (0-3)
  thal: 3 norma, 6 trwały ubytek, 7 odwracalny ubytek
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "dane", "heart_cleveland.csv")

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]   # ca: porządkowa 0-3 -> traktujemy liczbowo
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]                     # nominalne -> one-hot
TARGET = "target"


def load_raw(drop_duplicates=True):
    df = pd.read_csv(CSV)
    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_indices(y, fractions=(0.6, 0.2, 0.2), seed=42):
    """Podział warstwowy (zachowuje proporcje klas) na train / val / test."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in fractions]
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        cuts = np.cumsum([int(round(f * len(idx))) for f in fractions[:-1]])
        for k, chunk in enumerate(np.split(idx, cuts)):
            parts[k] += list(chunk)
    return [np.sort(np.array(p)) for p in parts]


class Preprocessor:
    """Imputacja braków + one-hot + standaryzacja. Statystyki liczone WYŁĄCZNIE na zbiorze
    treningowym (fit), a potem stosowane do val/test (transform) - brak wycieku danych."""

    def __init__(self, merge_rare_restecg=True, standardize=True, missing_indicators=False):
        """missing_indicators=True: dodatkowo kolumny 0/1 "czy brakowało wartości" (ca_brak, thal_brak)
        - popularna strategia, ale przy bardzo rzadkich brakach tworzy kolumny z 1-2 jedynkami."""
        self.merge_rare_restecg = merge_rare_restecg
        self.missing_indicators = missing_indicators
        self.standardize = standardize

    def _base(self, df):
        df = df.copy()
        if self.merge_rare_restecg:
            # restecg=1 (ST-T) ma tylko 4 próbki -> łączymy z restecg=2 w kategorię "nieprawidłowe EKG"
            df.loc[df.restecg == 1, "restecg"] = 2
        return df

    def fit(self, df):
        df = self._base(df)
        # braki: ca (liczba naczyń) i thal -> dominanta z treningu (obie cechy dyskretne,
        # braków jest bardzo mało, < 2% wierszy, więc prosta imputacja wystarcza)
        self.fill_ = {c: df[c].mode()[0] for c in ["ca", "thal"]}
        self.categories_ = {c: sorted(df[c].dropna().unique()) for c in CATEGORICAL}
        X = self._encode(df)
        self.mu_ = X.mean(0)
        self.sd_ = X.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        return self

    def _encode(self, df):
        cols, names = [], []
        if self.missing_indicators:
            for c in self.fill_:
                cols.append(df[c].isna().to_numpy(float)); names.append(f"{c}_brak")
        df = df.fillna(self.fill_)
        for c in NUMERIC + BINARY:
            cols.append(df[c].to_numpy(float)); names.append(c)
        for c in CATEGORICAL:
            # one-hot z pominięciem pierwszej kategorii (unikamy idealnej współliniowości z biasem)
            for v in self.categories_[c][1:]:
                cols.append((df[c] == v).to_numpy(float)); names.append(f"{c}_{int(v)}")
        self.feature_names_ = names
        return np.column_stack(cols)

    def transform(self, df):
        X = self._encode(self._base(df))
        if self.standardize:
            X = (X - self.mu_) / self.sd_
        return X

    def fit_transform(self, df):
        return self.fit(df).transform(df)


def load_splits(seed=42, **pre_kw):
    """Zwraca (Xtr, ytr), (Xva, yva), (Xte, yte), nazwy_cech."""
    df = load_raw()
    y = df[TARGET].to_numpy(float)
    tr, va, te = split_indices(y, seed=seed)
    pre = Preprocessor(**pre_kw).fit(df.iloc[tr])
    X = {k: pre.transform(df.iloc[i]) for k, i in (("tr", tr), ("va", va), ("te", te))}
    return (X["tr"], y[tr]), (X["va"], y[va]), (X["te"], y[te]), pre.feature_names_

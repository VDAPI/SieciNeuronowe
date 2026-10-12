---
title: "Ćwiczenie 1 – materiał do nauki: analiza eksploracyjna, preprocessing, baseline"
---

Ten plik jest do nauki przed wejściówką i obroną zadania. Kolejność: (1) o co chodzi, (2) dane,
(3) teoria, (4) kod linijka po linijce, (5) co wyszło, (6) pytania z odpowiedziami, (7) modyfikacje,
o które może poprosić prowadzący.

# 1. O co chodzi w ćwiczeniu

Zanim uczy się jakikolwiek model, trzeba **poznać dane** i **przygotować je** tak, żeby model mógł na nich pracować.
Treść zadania wymaga:

1. odpowiedzi na 5 pytań o dane (balans klas, średnie/odchylenia, normalność, równomierność kategorii, braki),
2. kodu, który **koduje** cechy kategoryczne i **normalizuje** dane (bo klasyfikatory liniowe i sieci neuronowe
   pracują na liczbach rzeczywistych i lubią dane o podobnej skali),
3. **baseline'u** – wyniku prostego, gotowego klasyfikatora ze scikit-learn, z którym porównamy własne modele w ćw. 2–6.

W ćw. 1–3 wolno używać bibliotek do danych i metryk, ale **metody uczenia w kolejnych ćwiczeniach trzeba napisać samemu**.

# 2. Zbiór Heart Disease (Cleveland)

303 pacjentów (1 duplikat → 302), 13 cech + klasa. Cel: przewidzieć, czy pacjent ma chorobę wieńcową
(zwężenie naczynia > 50%). Oryginalna kolumna `num` ma wartości 0–4; standardowo binaryzujemy: 0 = zdrowy, 1–4 = chory.

| cecha | znaczenie | typ |
|---|---|---|
| age | wiek | liczbowa |
| sex | płeć (1 = M, 0 = K) | binarna |
| cp | typ bólu w klatce: 1 typowa dławica, 2 nietypowa, 3 ból niedławicowy, 4 **bezobjawowo** | kategoryczna |
| trestbps | ciśnienie spoczynkowe [mm Hg] | liczbowa |
| chol | cholesterol [mg/dl] | liczbowa |
| fbs | cukier na czczo > 120 mg/dl | binarna |
| restecg | spoczynkowe EKG: 0 norma, 1 zmiany ST-T, 2 przerost lewej komory | kategoryczna |
| thalach | maksymalne tętno w teście wysiłkowym | liczbowa |
| exang | dławica wywołana wysiłkiem | binarna |
| oldpeak | obniżenie odcinka ST po wysiłku | liczbowa |
| slope | nachylenie odcinka ST: 1 wznoszące, 2 płaskie, 3 opadające | kategoryczna (porządkowa) |
| ca | liczba głównych naczyń widocznych we fluoroskopii (0–3) | liczbowa dyskretna / porządkowa |
| thal | talasemia / test talowy: 3 norma, 6 trwały ubytek, 7 odwracalny ubytek | kategoryczna |

Ciekawostka, o którą można zapytać: „bezobjawowy” ból (cp = 4) najsilniej wiąże się z chorobą (73% chorych).
To nie błąd danych, tylko cecha tej próby pacjentów (wszyscy byli kierowani na diagnostykę). Mówimy o korelacji, nie o przyczynie.

# 3. Teoria

## 3.1 Zbalansowanie klas

Zbiór jest **niezbalansowany**, gdy jedna klasa jest dużo liczniejsza (np. 95% / 5%). Wtedy model może „nauczyć się”
zawsze przewidywać klasę większościową i mieć wysokie accuracy, nie umiejąc nic. Tu: 164 / 138 (54% / 46%),
stosunek 0,84 → **praktycznie zbalansowany**, ale i tak warto patrzeć na precision/recall/F1, nie tylko accuracy.
Metody na niezbalansowanie (gdyby było): ważenie klas w funkcji kosztu, oversampling (np. SMOTE), undersampling,
zmiana progu decyzyjnego, podział warstwowy (stratified).

## 3.2 Statystyki opisowe

* **Średnia** μ = (1/N) Σ xᵢ, **odchylenie standardowe** σ = √((1/N) Σ (xᵢ − μ)²) – miara rozrzutu.
* **Mediana** – wartość środkowa, odporna na wartości odstające (chol: średnia 246,5, mediana 240,5 → prawy ogon).
* **Skośność** (skewness): 0 dla rozkładu symetrycznego, > 0 długi prawy ogon (chol 1,15; oldpeak 1,27), < 0 lewy (thalach −0,53).
* **Kurtoza** (nadwyżkowa): 0 dla normalnego, > 0 ciężkie ogony / wartości odstające (chol 4,5 – przez wartość 564).

## 3.3 Czy rozkład jest normalny?

Trzy sposoby, wszystkie są w kodzie:

1. **Histogram** z nałożoną krzywą N(μ, σ) – ocena „na oko”.
2. **Wykres Q-Q** (kwantyl-kwantyl): oś X – kwantyle teoretyczne rozkładu normalnego, oś Y – kwantyle danych.
   Punkty na prostej → rozkład normalny. Wygięcie w „banan” → skośność, odchylenia na końcach → ciężkie/lekkie ogony,
   „schodki” → dane dyskretne (ca).
3. **Test Shapiro-Wilka**: H₀ = „dane pochodzą z rozkładu normalnego”. Jeśli p < 0,05, odrzucamy H₀.
   Uwaga: przy dużej liczbie próbek test odrzuca normalność nawet przy drobnych odchyleniach, dlatego patrzymy
   też na wykresy. Tu `age` formalnie odpada (p = 0,007), ale na Q-Q wygląda prawie normalnie.

**p-wartość** = prawdopodobieństwo uzyskania tak skrajnego (lub bardziej) wyniku, gdyby H₀ była prawdziwa.
Mała p-wartość = dane słabo pasują do H₀. **Nie** jest to prawdopodobieństwo, że H₀ jest prawdziwa.

Po co nam to? Standaryzacja z-score nie wymaga normalności, ale przy silnie skośnych cechach (oldpeak, chol)
można rozważyć transformację log lub Box-Cox. Wartości odstające wpływają na μ i σ.

## 3.4 Czy rozkład kategorii jest równomierny?

Wykres słupkowy + **test χ² zgodności** (chi-kwadrat): H₀ = „każda kategoria ma tę samą liczność”.
Statystyka χ² = Σ (Oᵢ − Eᵢ)² / Eᵢ, gdzie O – obserwowane, E = N/k – oczekiwane. Tu wszystkie p < 10⁻⁸ → żadna cecha nie jest równomierna.
Dlaczego to ważne: bardzo rzadkie kategorie (restecg = 1: 4 próbki) dają kolumny one-hot prawie zawsze równe 0 –
model nie ma z czego się ich nauczyć, a w podziale na train/test mogą w ogóle nie trafić do jednego ze zbiorów.
Dlatego łączymy restecg = 1 z restecg = 2 („nieprawidłowe EKG”).

## 3.5 Braki danych

W oryginalnym pliku braki to `?`: **ca – 4, thal – 2** (6 wierszy, 2%).
Typy braków: **MCAR** (całkowicie losowe), **MAR** (zależne od innych obserwowanych cech), **MNAR** (zależne od samej brakującej wartości).
Strategie:

| strategia | plusy | minusy |
|---|---|---|
| usunięcie wierszy | proste | tracimy dane (mały zbiór!) |
| średnia / mediana (liczbowe) | proste, zachowuje N | zaniża wariancję |
| **dominanta** (kategoryczne / dyskretne) – **wybrana** | sensowna dla ca, thal | j.w. |
| kNN / model regresyjny | wykorzystuje zależności | bardziej złożone, ryzyko wycieku |
| osobna kategoria / kolumna-wskaźnik braku | model „wie”, że był brak | przy 1–2 brakach tworzy prawie pustą kolumnę (zob. ćw. 2, sekcja 6.2) |

Ważne: statystyki do imputacji (tu dominanta) liczymy **tylko na zbiorze treningowym**.

## 3.6 Kodowanie cech kategorycznych

* **Label encoding** (cp = 1, 2, 3, 4 jako liczba) jest zły dla cech **nominalnych**: model liniowy założy,
  że „4 jest dwa razy dalej od 2 niż 3” i że efekt rośnie liniowo – nie ma to sensu dla typu bólu.
* **One-hot encoding**: kategoria → osobna kolumna 0/1. Dla cp: cp_2, cp_3, cp_4.
* **Pułapka zmiennych sztucznych (dummy variable trap)**: jeśli zrobimy kolumnę dla każdej kategorii, ich suma = 1 =
  kolumna wyrazu wolnego → idealna współliniowość, wagi nie są jednoznaczne. Dlatego **pomijamy pierwszą kategorię**
  (staje się kategorią odniesienia, jej efekt „siedzi” w bias).
* Cechy binarne (sex, fbs, exang) już są 0/1. `ca` (0–3) jest porządkowa, więc zostawiamy ją jako liczbę.

## 3.7 Normalizacja i standaryzacja

* **Standaryzacja (z-score)**: x' = (x − μ) / σ → średnia 0, odchylenie 1. **Użyta w kodzie.**
* **Normalizacja min-max**: x' = (x − min) / (max − min) → przedział [0, 1]. Wrażliwa na wartości odstające.

Dlaczego to potrzebne dla modeli gradientowych (regresja logistyczna, sieci):

1. cechy o dużej skali (chol ~250) dominują sumę Wx i gradient; cechy małe (oldpeak ~1) prawie się nie uczą,
2. funkcja kosztu robi się „wąską doliną” (złe uwarunkowanie) – jeden współczynnik uczenia jest za duży dla jednego kierunku i za mały dla innego (w ćw. 2 to widać: bez standaryzacji uczenie praktycznie nie działa),
3. sigmoida/tanh nasycają się przy dużych wejściach → zanikający gradient,
4. wagi są porównywalne między cechami (interpretacja).

Modele drzewiaste (Random Forest) **nie** potrzebują normalizacji – dzielą po progach.

## 3.8 Wyciek danych (data leakage)

Jeśli μ, σ czy dominantę policzymy na całym zbiorze, informacja ze zbioru testowego „przecieka” do treningu
i wynik na teście jest zbyt optymistyczny. Poprawnie: **fit tylko na train, transform na train/val/test**.
Dokładnie tak działa klasa `Preprocessor` (`fit` → `transform`), a w walidacji krzyżowej preprocessing jest dopasowywany od nowa w każdym foldzie.

## 3.9 Podział danych i walidacja krzyżowa

* **train** (60%, 181) – uczenie wag, **val** (20%, 61) – dobór hiperparametrów / early stopping, **test** (20%, 60) – jednorazowa, końcowa ocena.
* **Podział warstwowy (stratified)** – w każdym podzbiorze ten sam odsetek klas (tu ~46% chorych).
* **k-krotna walidacja krzyżowa (k-fold CV)**: dzielimy dane na k części, k razy uczymy na k−1 i testujemy na 1, uśredniamy.
  Przy 60 próbkach testowych jedna próbka to ~1,7 pp accuracy, więc CV daje stabilniejszą ocenę (średnia ± odchylenie).

## 3.10 Metryki klasyfikacji

Macierz pomyłek (klasa pozytywna = 1, choroba):

|  | przewidziane 1 | przewidziane 0 |
|---|---|---|
| **faktycznie 1** | TP | FN |
| **faktycznie 0** | FP | TN |

* **accuracy** = (TP + TN) / N – odsetek trafień,
* **precision** = TP / (TP + FP) – „ile z oznaczonych jako chorzy jest naprawdę chorych”,
* **recall** (czułość) = TP / (TP + FN) – „ilu chorych wykryliśmy”; w medycynie zwykle najważniejszy (FN = przeoczony chory),
* **F1** = 2·P·R / (P + R) – średnia harmoniczna; niska, gdy którakolwiek z P, R jest niska,
* **ROC AUC** – pole pod krzywą ROC (TPR vs FPR dla wszystkich progów); 0,5 = losowo, 1 = idealnie; nie zależy od progu 0,5.

Dummy (zawsze 0) ma accuracy 0,55, ale F1 = 0 – dobra ilustracja, czemu accuracy nie wystarcza.

## 3.11 Baseline

Baseline to prosty punkt odniesienia. Jeśli nasz własny model w ćw. 2+ nie pobije (albo nie dorówna) prostej
regresji logistycznej ze sklearn, to coś jest nie tak. Mamy trzy poziomy: Dummy (dolna granica),
LogisticRegression (model liniowy) i RandomForest (model nieliniowy).

## 3.12 Korelacja Pearsona

r = cov(X, Y) / (σ_X σ_Y), od −1 do 1; mierzy tylko zależność **liniową**. Korelacja cechy z klasą 0/1 to tzw.
korelacja punktowo-dwuseryjna. Najsilniejsze: cp_4 (0,52), thal_7 (0,49), ca (0,46), exang (0,44), oldpeak (0,43), thalach (−0,42).

# 4. Kod krok po kroku

## 4.1 `data.py` – wczytanie i preprocessing (używane też w ćw. 2)

```python
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY = ["sex", "fbs", "exang"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]      # -> one-hot
```
Podział cech na typy decyduje o tym, co z nimi zrobimy.

```python
def load_raw(drop_duplicates=True):
    df = pd.read_csv(CSV)                 # puste pola -> NaN (braki)
    if drop_duplicates:
        df = df.drop_duplicates()         # 1 zduplikowany wiersz
```

`split_indices` – podział warstwowy: dla każdej klasy osobno losowa permutacja indeksów (`rng.permutation`)
i pocięcie na 60/20/20, potem sklejenie. Dzięki temu w każdym zbiorze jest ~46% chorych. Stały `seed=42` daje powtarzalność.

Klasa `Preprocessor` (wzorzec jak w sklearn: `fit` / `transform`):

* `_base` – łączy rzadkie restecg = 1 z restecg = 2,
* `fit(df_train)` – zapamiętuje: dominanty do imputacji (`self.fill_`), listę kategorii (`self.categories_`),
  średnie i odchylenia kolumn po zakodowaniu (`self.mu_`, `self.sd_`). `sd == 0` zamieniamy na 1, żeby nie dzielić przez zero,
* `_encode` – (opcjonalnie wskaźniki braków), `fillna(dominanta)`, cechy liczbowe i binarne bez zmian,
  dla kategorycznych `(df[c] == v)` dla każdej kategorii **poza pierwszą** → kolumny 0/1,
* `transform(df)` – kodowanie + standaryzacja statystykami **z treningu**.

`load_splits()` – składa wszystko: wczytaj → podziel → `fit` na train → `transform` train/val/test. Zwraca 17 cech.

## 4.2 `eda.py` – analiza

1. **Klasy**: `value_counts()` + wykres słupkowy.
2. **Liczbowe**: `describe()` (średnia, std, min, mediana, max), `skew()`, `kurt()`, `scipy.stats.shapiro` dla każdej cechy;
   wykresy: histogram + `stats.norm.pdf` (krzywa normalna o tych samych μ, σ), `stats.probplot` (Q-Q), boxploty w podziale na klasy.
3. **Kategoryczne**: `value_counts()`, `stats.chisquare(liczności)` (domyślnie porównuje z równomiernym),
   `groupby(c)[target].mean()` – odsetek chorych w każdej kategorii, `pd.crosstab` – wykres słupkowy skumulowany.
4. **Braki**: `df.isna().sum()` i wypisanie wierszy z brakami.
5. **Korelacje** po zakodowaniu: `np.corrcoef`, mapa ciepła.
6. **Baseline**: preprocessing dopasowany na train+val, trzy modele (`DummyClassifier`, `LogisticRegression`,
   `RandomForestClassifier`), metryki ze `sklearn.metrics` na teście oraz 5-krotna `StratifiedKFold` (w każdym foldzie nowy `Preprocessor`).
7. Wszystko zapisane do `results.json` i `figures/`.

# 5. Co wyszło (najważniejsze liczby)

* 302 próbki, 164 zdrowych / 138 chorych – prawie zbalansowany.
* Żadna cecha liczbowa nie jest normalna wg Shapiro-Wilka; najbliżej `age`, najdalej `oldpeak` i `ca`.
* Żadna cecha kategoryczna nie jest równomierna; restecg = 1 ma 4 próbki.
* Braki: ca 4, thal 2 → dominanta.
* Baseline: LogisticRegression F1 = 0,80 (test), 0,815 ± 0,026 (CV); RandomForest 0,78 / 0,813; Dummy 0.

# 6. Pytania na wejściówkę (z odpowiedziami)

1. **Kiedy zbiór jest niezbalansowany i czym to grozi?** Gdy klasy mają bardzo różne liczności; model może przewidywać tylko klasę większościową i mieć wysokie accuracy. Tu 54/46 – zbalansowany.
2. **Dlaczego accuracy nie wystarcza?** Przy niezbalansowaniu jest myląca; nie rozróżnia FP i FN. Dummy ma tu 55% accuracy przy F1 = 0.
3. **Zdefiniuj precision, recall, F1.** P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R).
4. **Która metryka jest ważniejsza w diagnostyce choroby serca?** Zwykle recall (koszt przeoczenia chorego jest wysoki), ale kosztem rosnącej liczby fałszywych alarmów.
5. **Co pokazuje wykres Q-Q?** Kwantyle danych vs kwantyle rozkładu normalnego; prosta linia = rozkład normalny.
6. **Co znaczy p < 0,05 w teście Shapiro-Wilka?** Odrzucamy hipotezę o normalności rozkładu na poziomie istotności 5%.
7. **Skośność dodatnia – co to znaczy?** Długi prawy ogon, średnia > mediana (chol, oldpeak).
8. **Dlaczego standaryzujemy dane przed regresją logistyczną / siecią?** Różne skale → złe uwarunkowanie, dominacja dużych cech w gradiencie, nasycenie funkcji aktywacji, wolna zbieżność.
9. **Standaryzacja a normalizacja min-max?** z-score: średnia 0, std 1; min-max: zakres [0,1], wrażliwa na wartości odstające.
10. **Czy standaryzacja zmienia kształt rozkładu?** Nie – to przesunięcie i przeskalowanie; skośność zostaje.
11. **Czym jest one-hot encoding i po co go używamy?** Kategoria → kolumny 0/1; nie narzuca sztucznego porządku i odległości między kategoriami.
12. **Dlaczego pomijamy jedną kategorię (drop first)?** Suma wszystkich kolumn one-hot = 1 = kolumna biasu → idealna współliniowość (dummy variable trap).
13. **Dlaczego nie kodujemy cp jako 1–4?** Cecha nominalna; model liniowy założyłby liniowy wpływ i równe odległości.
14. **Jakie są strategie uzupełniania braków?** Usunięcie, średnia/mediana/dominanta, kNN, model, osobna kategoria/wskaźnik. Wybraliśmy dominantę (mało braków, cechy dyskretne).
15. **Co to jest wyciek danych?** Użycie informacji ze zbioru testowego podczas uczenia/preprocessingu (np. μ i σ z całego zbioru) → zawyżone wyniki.
16. **Po co zbiór walidacyjny, skoro mamy testowy?** Walidacyjny służy do doboru hiperparametrów i zatrzymania uczenia; testowy tylko do końcowej, uczciwej oceny.
17. **Co to jest podział warstwowy?** Zachowuje proporcje klas w każdym podzbiorze.
18. **Na czym polega k-fold CV?** k podziałów, każda część raz jest testowa, wynik uśredniony – stabilniejsza ocena na małych zbiorach.
19. **Co mierzy ROC AUC?** Jakość rankingu prawdopodobieństw niezależnie od progu; 0,5 losowo, 1 idealnie.
20. **Czemu Random Forest nie potrzebuje normalizacji?** Drzewa dzielą po progach pojedynczej cechy; skala nie ma znaczenia.
21. **Co to jest baseline?** Wynik prostego modelu referencyjnego, do którego porównujemy własne rozwiązania.
22. **Korelacja Pearsona – co mierzy i czego nie?** Siłę zależności liniowej; nie wykrywa zależności nieliniowych i nie oznacza przyczynowości.
23. **Które cechy najbardziej wiążą się z chorobą?** cp = 4 (bezobjawowy), thal = 7, ca, exang, oldpeak, thalach (ujemnie).
24. **Co zrobić z bardzo rzadką kategorią?** Połączyć z podobną (restecg 1+2), albo zostawić świadomie; prawie pusta kolumna niewiele wnosi i destabilizuje uczenie.

# 7. Modyfikacje, o które może poprosić prowadzący

* **Normalizacja min-max zamiast z-score** – w `Preprocessor.fit` zapamiętaj `self.min_ = X.min(0)`, `self.max_ = X.max(0)`,
  a w `transform`: `X = (X - self.min_) / (self.max_ - self.min_)`.
* **Imputacja medianą / usuwanie wierszy z brakami** – zmień `self.fill_ = {c: df[c].median() ...}` albo w `load_raw` dodaj `df = df.dropna()`.
* **Inny podział (np. 80/20, bez walidacji)** – `split_indices(y, fractions=(0.8, 0.2))` zwróci dwie listy indeksów.
* **Inny baseline** (np. SVM, kNN) – dopisz do słownika `models`: `"SVC": SVC(probability=True)` (`from sklearn.svm import SVC`).
* **Transformacja log skośnej cechy** – przed kodowaniem: `df["oldpeak"] = np.log1p(df["oldpeak"])`.
* **One-hot bez pomijania kategorii** – w `_encode` zamień `self.categories_[c][1:]` na `self.categories_[c]` (i wyjaśnij, czemu to tworzy współliniowość).

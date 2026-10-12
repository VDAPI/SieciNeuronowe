---
title: "Ćwiczenie 2 – materiał do nauki: regresja logistyczna i spadek gradientu"
---

Plik do nauki przed wejściówką i obroną. Kolejność: (1) o co chodzi, (2) teoria z wyprowadzeniami,
(3) kod linijka po linijce, (4) jak czytać wyniki eksperymentów, (5) pytania z odpowiedziami,
(6) modyfikacje kodu, o które może poprosić prowadzący.

# 1. O co chodzi w ćwiczeniu

Trzeba **samodzielnie zaimplementować** regresję logistyczną (bez gotowego `fit` ze sklearn) i uczyć ją
spadkiem gradientu na zbiorze Heart Disease. Potem zbadać:

* jakość klasyfikacji: **precision, recall, F1**,
* szybkość zbieżności w zależności od: **współczynnika uczenia**, **rozkładu inicjalizacji wag**, **liczby epok / warunku stopu**,
* narzędzie: **krzywa uczenia** – koszt i metryki w funkcji epoki.

Pytanie badawcze z treści: *czy klasyfikator liniowy zawsze zbiega do globalnego minimum? Czy zła parametryzacja
uniemożliwia zbieżność / dobrą jakość, czy tylko przedłuża uczenie?*

Regresja logistyczna = **jeden neuron z sigmoidalną funkcją aktywacji**. Wszystko, co tu poznasz (koszt, gradient,
współczynnik uczenia, inicjalizacja, krzywa uczenia, early stopping), wraca w ćw. 3 przy sieciach wielowarstwowych –
tam jednak funkcja kosztu **nie** jest wypukła.

# 2. Teoria

## 2.1 Model

z = Wx + b = w₁x₁ + … + w_d x_d + b  (logit),  p(x) = σ(z) = 1 / (1 + e^(−z)).

* p(x) interpretujemy jako P(y = 1 | x) – prawdopodobieństwo choroby.
* Decyzja: ŷ = 1, gdy p ≥ 0,5 ⇔ z ≥ 0. **Granica decyzyjna Wx + b = 0 to hiperpłaszczyzna** → klasyfikator liniowy.
* Logit = log-szansa: ln(p / (1 − p)) = Wx + b. Waga wᵢ mówi, o ile rośnie log-szansa, gdy xᵢ rośnie o 1 (po standaryzacji: o 1 odchylenie standardowe).

## 2.2 Sigmoida – własności

* wartości w (0, 1), rosnąca, σ(0) = 0,5, σ(−z) = 1 − σ(z),
* **pochodna: σ'(z) = σ(z)(1 − σ(z))**, maksymalnie 0,25 (w z = 0), → 0 dla dużych |z| (**nasycenie**),
* stabilność numeryczna: dla dużych ujemnych z `exp(-z)` przepełnia się. W kodzie dla z < 0 używamy równoważnej postaci e^z / (1 + e^z).

Wyprowadzenie pochodnej: σ = (1 + e^(−z))⁻¹ → σ' = e^(−z) / (1 + e^(−z))² = σ · (e^(−z) / (1 + e^(−z))) = σ(1 − σ).

## 2.3 Funkcja kosztu – entropia krzyżowa (binary cross-entropy, BCE)

L = −[y ln p + (1 − y) ln(1 − p)], w kodzie **średnia** po N przykładach.

* y = 1 → L = −ln p: kara rośnie do ∞, gdy p → 0 (pewna, błędna odpowiedź),
* y = 0 → L = −ln(1 − p).

**Skąd ten wzór (metoda największej wiarygodności):** y ma rozkład Bernoulliego z parametrem p:
P(y | x) = p^y (1 − p)^(1−y). Wiarygodność danych = Π P(yᵢ | xᵢ). Maksymalizacja wiarygodności = minimalizacja
ujemnego logarytmu = Σ BCE. Czyli BCE nie jest wymyślona „z sufitu”.

**Dlaczego nie MSE (błąd średniokwadratowy)?**
1. MSE złożone z sigmoidą **nie jest wypukłe** w W,
2. gradient MSE zawiera σ'(z) → przy nasyceniu (pewna, ale błędna odpowiedź) gradient ≈ 0 i model przestaje się uczyć.
   W BCE σ' się skraca (patrz niżej) – im większy błąd, tym większy gradient.

Wartość odniesienia: dla p = 0,5 (wagi zerowe) L = ln 2 ≈ **0,693** – to jest koszt początkowy w eksperymentach.

## 2.4 Wyprowadzenie gradientu (reguła łańcuchowa) – częste pytanie!

∂L/∂wᵢ = ∂L/∂p · ∂p/∂z · ∂z/∂wᵢ

* ∂L/∂p = −y/p + (1 − y)/(1 − p) = (p − y) / (p(1 − p)),
* ∂p/∂z = p(1 − p),
* ∂z/∂wᵢ = xᵢ.

Mnożymy: **∂L/∂wᵢ = (p − y) xᵢ = −(y − p) xᵢ**, a dla biasu ∂L/∂b = p − y.
Czynnik p(1 − p) się skraca – to jest ta „wygoda” BCE z treści zadania.

Wektorowo dla całego zbioru (średnia): **∇_W L = (1/N) Xᵀ(p − y)**, ∂L/∂b = mean(p − y).
Interpretacja: gradient = suma wektorów cech ważona błędem predykcji.

## 2.5 Spadek gradientu (gradient descent)

w ← w − α ∇L. Gradient wskazuje kierunek najszybszego **wzrostu** funkcji, więc idziemy w przeciwną stronę.

* **α (learning rate)** – hiperparametr, długość kroku,
* **epoka** = jedno przejście przez cały zbiór treningowy; tu (pełny batch) 1 epoka = 1 krok,
* **batch GD** (cały zbiór, używany tu – deterministyczny, gładka krzywa), **SGD** (1 próbka na krok – szum, tanie kroki),
  **mini-batch** (np. 32 próbki – standard w sieciach).

## 2.6 Wypukłość – dlaczego regresja logistyczna jest „łatwa”

Funkcja f jest wypukła, gdy f(λa + (1−λ)b) ≤ λf(a) + (1−λ)f(b) – odcinek między dowolnymi punktami wykresu leży nad wykresem.
Dla funkcji dwukrotnie różniczkowalnej: **hesjan dodatnio półokreślony**.

Hesjan BCE: H = (1/N) Xᵀ S X, gdzie S = diag(pᵢ(1 − pᵢ)) ≥ 0 → vᵀHv = (1/N) Σ pᵢ(1−pᵢ)(xᵢᵀv)² ≥ 0 → **wypukła**.

Konsekwencje: **każde minimum lokalne jest globalne**, brak minimów lokalnych, w które można „wpaść”, więc spadek
gradientu (z odpowiednio małym α) zawsze zmierza do minimum globalnego, niezależnie od punktu startowego.

**Ale są dwa wyjątki** (ważne dla pytania badawczego):

1. **Dane liniowo separowalne** (w całości lub w jakimś kierunku): koszt można zmniejszać w nieskończoność, zwiększając ‖w‖ → minimum
   **nie jest osiągane dla skończonych wag** (infimum = 0 „w nieskończoności”). Koszt zbiega, wagi rosną bez końca.
   W raporcie, sekcja 6.2: kolumna `thal_brak` z jedną jedynką → waga `thal_brak` rośnie (co do modułu) jak log(epoki).
   Lekarstwo: regularyzacja L2 (wtedy funkcja jest ściśle wypukła i minimum istnieje), lepszy preprocessing, early stopping.
2. **Współliniowość cech** (np. one-hot bez pominięcia kategorii): hesjan jest osobliwy → minimum istnieje, ale **nie jest jednoznaczne**
   (cała prosta/płaszczyzna rozwiązań o tym samym koszcie).

## 2.7 Współczynnik uczenia a stabilność – skąd próg?

Lokalnie (blisko minimum) koszt wygląda jak funkcja kwadratowa z hesjanem H. Spadek gradientu na funkcji kwadratowej
jest stabilny ⇔ **α < 2 / λ_max(H)**, a zbieżność w najwolniejszym kierunku zależy od α·λ_min.
Liczba kroków ~ **κ = λ_max / λ_min** (wskaźnik uwarunkowania).

Policzone dla naszych danych (zbiór treningowy, standaryzowany):

* w minimum: λ_max ≈ 0,195 → **2/λ_max ≈ 10,3**. Zgadza się z eksperymentem: α = 5 zbiega (14 epok), α = 20 oscyluje i nie zbiega.
* w punkcie startowym w = 0: λ_max ≈ 0,89 → 2/λ_max ≈ 2,2. α = 5 jest „za duże” na starcie, ale BCE nie jest kwadratowa,
  krzywizna maleje, gdy p oddala się od 0,5, więc mimo to się udaje.
* dane **niestandaryzowane**: λ_max ≈ 26 700 → 2/λ_max ≈ **7,5·10⁻⁵**, a κ ≈ 2,8·10⁷.
  To tłumaczy wyniki: α = 10⁻³ i 10⁻² rozbiegają się/oscylują, α = 10⁻⁵ i 10⁻⁴ są stabilne, ale po 20 000 epokach koszt jest wciąż daleko od minimum.

Podsumowanie: **za małe α** – powolna zbieżność (ale zbieżność); **dobre α** – szybko; **za duże α** – oscylacje / rozbieżność.

## 2.8 Standaryzacja a kształt funkcji kosztu

Bez standaryzacji poziomice kosztu to bardzo wydłużone elipsy (wąska dolina). Gradient jest prostopadły do poziomic,
więc wskazuje „w poprzek doliny”, a nie wzdłuż niej → zygzaki. Standaryzacja zbliża poziomice do okręgów (κ z ~5·10⁶ do ~38 dla XᵀX).
Minimum jest to samo (standaryzacja to przekształcenie afiniczne; wagi się przeskalowują), zmienia się tylko łatwość jego znalezienia.

## 2.9 Inicjalizacja

* **W regresji logistycznej zera są w porządku**: jeden neuron, nie ma problemu symetrii; startujemy z p = 0,5, L = ln 2.
* **W sieciach wielowarstwowych (ćw. 3) zera są złe**: wszystkie neurony w warstwie dostają ten sam gradient i pozostają identyczne (problem symetrii). Dlatego tam losujemy (Xavier/Glorot, He).
* **Duże wagi początkowe** (σ = 10, stała 5): duże |z| → sigmoida nasycona → p ≈ 0 lub 1, koszt startowy ogromny (do ~10).
  Dla błędnie sklasyfikowanych próbek gradient jest ograniczony (|p − y| ≤ 1), więc „wyjście” z takiego punktu trwa długo.
  Dzięki wypukłości wynik końcowy jest ten sam – to tylko **wydłuża** uczenie (N(0;10) nie zdążyło w 5000 epok w żadnym z 10 przebiegów).
* Z małymi inicjalizacjami różnice w liczbie epok są niewielkie (890–985 epok).

## 2.10 Warunki stopu i krzywa uczenia

* **stała liczba epok** – proste, ale nie wiadomo, ile wybrać,
* **|L_t − L_(t−1)| < tol** – stop, gdy koszt przestał się zmieniać (lub ‖∇L‖ < tol),
* **early stopping** – monitorujemy koszt na **walidacji**; jeśli nie poprawia się przez `patience` epok, stop i przywracamy najlepsze wagi.

**Krzywa uczenia** = koszt/metryki w funkcji epoki dla train i walidacji. Jak czytać:

* obie maleją → model się uczy,
* train maleje, walidacja rośnie → **przeuczenie (overfitting)**; u nas koszt walidacyjny ma minimum ok. epoki 111 i potem rośnie (0,386 → 0,433),
  bo wagi rosną, a predykcje stają się „zbyt pewne”,
* obie wysokie i płaskie → **niedouczenie** (model za prosty albo α za małe).

Uwaga: minimum kosztu **treningowego** (cel optymalizacji) to nie to samo co najlepszy model (minimum błędu **uogólnienia**).
Remedium: early stopping albo regularyzacja L2 (L + λ‖w‖²/2).

**F1 a koszt:** F1 zależy tylko od tego, po której stronie progu 0,5 jest p, więc jest „schodkowe” i może stać w miejscu,
gdy koszt dalej spada (lub rośnie na walidacji). Dlatego optymalizujemy BCE (różniczkowalna), a raportujemy F1.

# 3. Kod krok po kroku (`logreg.py`)

```python
def sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))      # dla z>=0: exp(-z) <= 1, bez przepełnienia
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)                   # dla z<0: ta sama funkcja, inna postać
    return out
```

```python
def bce(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)                  # ln(0) = -inf -> obcinamy
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
```

`metrics` liczy TP, FP, FN, TN „ręcznie” i z nich precision, recall, F1, accuracy (próg 0,5).

`init_params(d, kind, scale)` – wagi startowe: `zeros`, `normal` N(0, scale), `uniform` U(−scale, scale), `const` = scale; bias = 0.

Klasa `LogisticRegressionGD`, metoda `fit` – serce zadania:

```python
for ep in range(self.epochs):
    p = self.predict_proba(X)            # p = sigmoid(X @ w + b), wektor N prawdopodobieństw
    err = y - p
    gw = -(X.T @ err) / N                # dL/dw = -(1/N) X^T (y - p)
    gb = -np.mean(err)                   # dL/db
    ... logowanie: koszt, F1/P/R na train i val, norma gradientu ...
    ... warunki stopu ('tol' albo 'early') ...
    self.w -= self.lr * gw               # krok gradientowy
    self.b -= self.lr * gb
```

* logowanie jest **przed** krokiem, więc `history["loss"][0]` to koszt dla wag początkowych,
* `stop="tol"`: porównuje dwa ostatnie koszty,
* `stop="early"`: pamięta `best = (najlepszy koszt val, wagi, bias, epoka)`; gdy przez `patience` epok nie ma poprawy – przywraca najlepsze wagi i kończy.

`epochs_to(loss_hist, target)` – pierwsza epoka, gdy koszt ≤ target. Target = L\* + 10⁻³ (nasza definicja „zbieżności”).

`main()` – eksperymenty:

1. **Minimum odniesienia** L\*: α = 0,5, do |ΔL| < 10⁻¹⁴ (+ porównanie ze sklearn – zgodność wag do 10⁻⁵).
2. **Baseline**: α = 0,1, zera, 2000 epok – krzywe uczenia, wagi.
3. **α** ∈ {10⁻⁴ … 100}, 5000 epok.
4. **Inicjalizacja**: 7 wariantów × 10 ziaren (dla losowych), średnia i zakres kosztu.
5. **Warunki stopu**: 10/100/1000/20000 epok, tol 10⁻⁴ i 10⁻⁶, early stopping.
6. **Brak standaryzacji** (`load_data(standardize=False)`).
7. **Wskaźniki braków** (`load_data(clean=False)`) – śledzenie ‖w‖ i wag przez 200 000 epok.

`data.py` – ten sam preprocessing co w ćw. 1 (patrz NAUKA_cw1): imputacja dominantą, one-hot, standaryzacja statystykami z treningu, podział 60/20/20.

# 4. Wyniki w pigułce – jak je czytać

| czynnik | co się dzieje | uniemożliwia czy wydłuża? |
|---|---|---|
| α za małe (≤ 0,01) | koszt monotonicznie, ale wolno spada | **wydłuża** |
| α 0,1–5 | zbieżność do L\* w 922 → 14 epok | – |
| α za duże (≥ 20 > 2/λ_max ≈ 10) | oscylacje, brak zbieżności | **uniemożliwia** (dla tego α) |
| duża inicjalizacja | nasycenie sigmoidy, start z L ≈ 8–10, setki–tysiące epok więcej | **wydłuża** |
| brak standaryzacji | złe uwarunkowanie (κ ~ 10⁷): dla stabilnego α praktycznie stoi | w praktyce **uniemożliwia** (teoretycznie wydłuża) |
| dane separowalne w jakimś kierunku | koszt zbiega, wagi → ∞ | minimum **nie istnieje** dla skończonych wag |
| długi trening | koszt train → L\*, koszt val rośnie | przeuczenie, F1 bez zmian |

Jakość: F1 = 0,80 na teście (P = 0,786, R = 0,815), identycznie jak sklearn → poprawna implementacja.

# 5. Pytania na wejściówkę (z odpowiedziami)

1. **Napisz model regresji logistycznej.** p(x) = σ(Wx + b), σ(z) = 1/(1 + e^(−z)).
2. **Dlaczego to „jednowarstwowa sieć neuronowa”?** Jeden neuron: suma ważona wejść + bias, aktywacja sigmoidalna.
3. **Pochodna sigmoidy?** σ'(z) = σ(z)(1 − σ(z)); maksimum 0,25 w z = 0.
4. **Zapisz entropię krzyżową.** L = −y ln p − (1 − y) ln(1 − p).
5. **Skąd się bierze BCE?** Z minimalizacji ujemnej log-wiarygodności rozkładu Bernoulliego.
6. **Wyprowadź ∂L/∂wᵢ.** Reguła łańcuchowa: (p − y)/(p(1−p)) · p(1−p) · xᵢ = (p − y)xᵢ.
7. **Dlaczego nie MSE?** Niewypukłe z sigmoidą; gradient zawiera σ', który znika przy nasyceniu.
8. **Reguła aktualizacji wag?** wᵢ ← wᵢ − α ∂L/∂wᵢ.
9. **Co to jest α i jak wpływa na uczenie?** Długość kroku; za małe – wolno, za duże – oscylacje/rozbieżność.
10. **Jaki jest teoretyczny próg stabilności α?** α < 2/λ_max(H); u nas ≈ 10 w pobliżu minimum.
11. **Co to znaczy, że funkcja jest wypukła i co z tego wynika?** Odcinek nad wykresem / hesjan ≥ 0; każde minimum lokalne jest globalne.
12. **Udowodnij wypukłość BCE dla regresji logistycznej.** H = (1/N)XᵀSX, S = diag(p(1−p)) ≥ 0 → vᵀHv ≥ 0.
13. **Czy regresja logistyczna zawsze osiąga minimum globalne?** Przy rozsądnym α tak, jeśli minimum istnieje. Nie istnieje (dla skończonych wag), gdy dane są liniowo separowalne; nie jest jednoznaczne przy współliniowości.
14. **Co się dzieje przy danych liniowo separowalnych?** ‖w‖ → ∞, koszt → 0 (lub do infimum), wagi nie zbiegają; pomaga regularyzacja L2.
15. **Czy można inicjalizować wagi zerami?** W regresji logistycznej tak (brak symetrii do złamania); w MLP nie.
16. **Dlaczego duże wagi początkowe spowalniają uczenie?** Nasycenie sigmoidy, ogromny koszt startowy, ograniczony gradient |p − y| ≤ 1.
17. **Co to jest epoka? Czym różni się od iteracji?** Epoka – przejście przez cały zbiór; iteracja – jedna aktualizacja wag. W batch GD 1 epoka = 1 iteracja.
18. **Batch GD vs SGD vs mini-batch?** Cały zbiór / 1 próbka / mała paczka na krok: dokładność gradientu vs koszt kroku i szum.
19. **Co to jest krzywa uczenia i co z niej odczytasz?** Koszt/metryki vs epoka; zbieżność, przeuczenie (val rośnie), niedouczenie.
20. **Jakie warunki stopu znasz?** Stała liczba epok, |ΔL| < tol, ‖∇L‖ < tol, early stopping na walidacji.
21. **Na czym polega early stopping?** Zatrzymanie, gdy koszt walidacyjny nie poprawia się przez `patience` epok, i przywrócenie najlepszych wag.
22. **Czy model liniowy może się przeuczyć?** Tak – u nas koszt walidacyjny rośnie po ~110 epokach, bo wagi rosną i predykcje są zbyt pewne.
23. **Dlaczego F1 może się nie zmieniać, gdy koszt spada?** F1 zależy tylko od progu 0,5; koszt także od pewności predykcji.
24. **Dlaczego standaryzacja przyspiesza uczenie?** Poprawia uwarunkowanie (κ), pozwala na większe α wspólne dla wszystkich kierunków.
25. **Ile wynosi koszt dla wag zerowych?** ln 2 ≈ 0,693 (bo p = 0,5 dla każdej próbki).
26. **Jak interpretować wagę?** Zmiana log-szansy choroby przy wzroście cechy o 1 (o 1σ po standaryzacji); znak = kierunek wpływu.
27. **Jak sprawdzić, że implementacja jest poprawna?** Porównać koszt i wagi ze sklearn (bez regularyzacji), sprawdzić ‖∇L‖ → 0, ewentualnie gradient numeryczny (różnice skończone).
28. **Odpowiedź na pytanie badawcze w jednym zdaniu?** (Sformułuj samodzielnie – patrz tabela w sekcji 4: które problemy tylko wydłużają uczenie, a które ją blokują.)

# 6. Modyfikacje, o które może poprosić prowadzący (gotowe fragmenty)

**Regularyzacja L2** (w `fit`, po policzeniu gradientu; bias zwykle bez regularyzacji):
```python
gw = -(X.T @ err) / N + self.l2 * self.w
# i do kosztu: bce(...) + 0.5 * self.l2 * np.sum(self.w ** 2)
```

**Mini-batch SGD** (zamiast jednego kroku na epokę):
```python
idx = self.rng.permutation(N)
for s in range(0, N, batch):
    B = idx[s:s + batch]
    pB = self.predict_proba(X[B]); eB = y[B] - pB
    self.w -= self.lr * (-(X[B].T @ eB) / len(B))
    self.b -= self.lr * (-np.mean(eB))
```

**Momentum**:
```python
self.v = beta * self.v + gw          # self.v = np.zeros_like(w) przed pętlą
self.w -= self.lr * self.v
```

**Malejący współczynnik uczenia**: `lr_t = self.lr / (1 + decay * ep)`.

**Stop na normie gradientu**: `if np.sqrt(gw @ gw + gb * gb) < self.tol: break`.

**Inny próg decyzyjny** (np. wyższy recall): `metrics(y, p, thr=0.3)`.

**Sprawdzenie gradientu numerycznie**:
```python
eps = 1e-6; i = 0; w0 = m.w.copy()
m.w[i] += eps; Lp = bce(y, m.predict_proba(X)); m.w[i] -= 2 * eps; Lm = bce(y, m.predict_proba(X)); m.w = w0
print((Lp - Lm) / (2 * eps), gw[i])   # powinny być prawie równe
```

**Koszt MSE dla porównania**: L = mean((p − y)²), gradient: `gw = (2/N) * X.T @ ((p - y) * p * (1 - p))` – zobaczysz wolniejsze uczenie przy nasyceniu.

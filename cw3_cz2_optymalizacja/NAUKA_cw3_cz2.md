---
title: "Ćwiczenie 3, cz. 2 – materiał do nauki: optymalizacja sieci, ADAM, inicjalizacja"
---

Plik do nauki przed wejściówką i obroną. Kolejność: (1) o co chodzi, (2) teoria z wyprowadzeniami,
(3) kod linijka po linijce, (4) jak czytać wyniki eksperymentów, (5) pytania z odpowiedziami,
(6) modyfikacje kodu, o które może poprosić prowadzący.

# 1. O co chodzi w ćwiczeniu

Do sieci z cz. 1 (moduły Linear / ReLU / sigmoida + BCE z `forward` i `backward`) trzeba **samodzielnie dopisać**:

* uczenie w trzech trybach: **pełny zbiór** (batch), **paczki** (mini-batch), **pojedyncze przykłady** (SGD),
* optymalizator **ADAM**,
* wybór **inicjalizacji wag**.

I porównać eksperymentalnie:

* pełny zbiór vs pojedyncze przykłady vs paczki,
* inicjalizację z odchyleniem 1 vs **He** vs „zepsutą” (bardzo duże odchylenie),
* prosty gradient vs ADAM,

dobierając głębokość sieci i warunki stopu tak, żeby **problemy optymalizacji było widać**. U nas: 5 warstw ukrytych
po 17 neuronów (zbiór wymaga najwyżej jednej), stała liczba epok, zapamiętywanie wag z najlepszej epoki na walidacji.

# 2. Teoria

## 2.1 Optymalizacja niewypukła: punkty siodłowe i płaskowyże

* W ćw. 2 koszt był wypukły → jedno minimum, gradient zawsze prowadzi do celu.
* Sieć z ≥ 2 warstwami → koszt niewypukły. Intuicja: „utkniemy w minimum lokalnym”. Wyniki z ostatnich 20 lat:
  w wielu wymiarach minima lokalne są zwykle prawie tak dobre jak globalne, a **głównym problemem są punkty
  siodłowe i płaskowyże** – miejsca, gdzie gradient ≈ 0, choć to nie jest minimum.
* **Punkt siodłowy**: gradient = 0, ale w jednych kierunkach funkcja rośnie, w innych maleje (hesjan ma wartości
  własne obu znaków). W d wymiarach losowy punkt krytyczny ma ogromną szansę być siodłem, nie minimum.
* **Płaskowyż**: duży obszar z gradientem ≈ 0 (np. okolica w = 0 przy małej inicjalizacji – cz. 1).
* Prosty GD: krok = α·g, więc przy g ≈ 0 krok ≈ 0 → „pełzanie”. Przykład z raportu: f = x² + y⁴/4 − y²/2,
  siodło w (0, 0); start (1; 10⁻⁴); w kierunku y gradient ≈ −y, więc |y| rośnie ok. 10% na krok → **91 kroków** do
  ucieczki. ADAM: **6 kroków**.

## 2.2 Tryby uczenia: batch, mini-batch, SGD

| tryb | gradient z | kroków na epokę (N = 181) | cechy |
|---|---|---|---|
| pełny zbiór (batch GD) | wszystkich N przykładów | 1 | dokładny, gładka krzywa, wolny w epokach |
| paczki (mini-batch) | B przykładów (u nas 32) | ⌈N/B⌉ = 6 | szum umiarkowany, wydajne mnożenie macierzy |
| pojedyncze (SGD) | 1 przykładu | 181 | bardzo zaszumiony, dużo kroków, wolny w czasie |

* **Epoka** = przejście przez cały zbiór; **krok (iteracja)** = jedna aktualizacja wag.
* Gradient z paczki to **nieobciążony estymator** pełnego gradientu (średnia pełnego = średnia gradientów pojedynczych;
  sprawdzone w testach). Wariancja estymatora maleje jak 1/B.
* Szum pomaga: nie liczymy dokładnej funkcji kosztu, tylko jej przybliżenie, więc łatwiej wypaść z płaskiego obszaru
  albo punktu siodłowego. Przed każdą epoką **losowa permutacja** przykładów.
* Wyniki: koszt ≤ 0,3 po 398 / 66 / 4 epokach, ale po **podobnej liczbie kroków** (398 / 398 / 724). W czasie: paczki
  najszybsze; 1000 epok SGD trwa ok. 15 s zamiast 0,3 s (181 małych operacji zamiast jednej dużej).

## 2.3 Momentum

v ← β·v + g,  w ← w − α·v (wersja „klasyczna”) albo m ← β·m + (1 − β)·g (średnia krocząca, jak w ADAM).

* Krok zachowuje **kierunek poprzednich kroków** (rozpęd) → przez punkt siodłowy przechodzimy „z rozpędu”,
  w wąskiej dolinie oscylacje w poprzek się znoszą, a ruch wzdłuż doliny się sumuje.
* β = 0,9 → średnia z ok. 1/(1 − β) = 10 ostatnich gradientów.

## 2.4 Skalowanie odwrotnością wielkości gradientu (RMSProp)

v ← β₂·v + (1 − β₂)·g²,  w ← w − α·g / (√v + ε).

* v = średnia krocząca **kwadratu** gradientu, osobno dla każdej wagi. Krok jest dzielony przez typową wielkość gradientu:
  gdy ostatnie kroki były małe (płaskowyż) → krok rośnie; gdy duże → maleje.
* Efekt: krok ma długość ~α **niezależnie od skali gradientu** – w raporcie gradient 10⁻⁸ (inicjalizacja N(0; 0,01))
  był dla ADAM tak samo dobry jak gradient ~1.
* Każda waga ma własny „współczynnik uczenia” → działa też przy złym uwarunkowaniu.

## 2.5 ADAM = momentum + skalowanie

Dla każdej wagi:

* m_t = β₁ m_{t−1} + (1 − β₁) g_t (pierwszy moment – średnia gradientu),
* v_t = β₂ v_{t−1} + (1 − β₂) g_t² (drugi moment – średnia kwadratu),
* **korekcja biasu**: m̂_t = m_t / (1 − β₁ᵗ), v̂_t = v_t / (1 − β₂ᵗ),
* w_t = w_{t−1} − α · m̂_t / (√v̂_t + ε).

Domyślnie β₁ = 0,9, β₂ = 0,999, ε = 10⁻⁸, α = 0,001.

**Po co korekcja biasu?** m i v startują od 0. Bez korekcji w pierwszym kroku m₁ = 0,1·g, v₁ = 0,001·g² – obie wartości
są „przesunięte” w stronę zera, i to różnie mocno. Po korekcji m̂₁ = g, v̂₁ = g², więc pierwszy krok to
α·g/|g| = **α·znak(g)** (sprawdzone w testach dla g od 10⁻³ do 10⁶). Z czasem 1 − βᵗ → 1 i korekcja znika.

**Po co ε?** Żeby nie dzielić przez zero, gdy v̂ ≈ 0. Gdy |g| jest porównywalne z ε, krok się zmniejsza.

**Wady ADAM:** może szybciej się przeuczać (szybko dochodzi do kosztu 10⁻⁵–10⁻⁷ w raporcie), wymaga warunku stopu;
za duże α też potrafi zepsuć uczenie (w prototypie ADAM α = 0,01 w sieci z 8 warstwami „zabił” neurony ReLU).

## 2.6 Inicjalizacja i wariancja sygnału

Dla y = Wx (bez biasu), W i x niezależne, E[W] = 0: **Var(yᵢ) = d_in · Var(W) · E[x²]**.

* Po ReLU E[h²] = Var(z)/2 (połowa wartości zerowana) → wariancja przechodzi przez warstwę z czynnikiem
  **d_in · σ² / 2**.
* Żeby była stała: d_in·σ²/2 = 1 → **σ = √(2/d_in)** – to jest **inicjalizacja He**. Ten sam czynnik działa wstecz,
  więc gradient też nie zanika ani nie eksploduje.
* U nas d_in = 17:
  * σ = 0,01 → czynnik 17·10⁻⁴/2 ≈ 10⁻³ dla wariancji, czyli ok. **30× mniej odchylenia na warstwę** → zanikanie
    (std wyjścia 4·10⁻¹⁰, gradient 10⁻⁸),
  * σ = 1 → czynnik 8,5 dla wariancji, ok. **3× więcej odchylenia na warstwę** → eksplozja (std 360, gradient ~100),
  * σ = 10 → ok. **30× więcej na warstwę** → std 4·10⁸, gradient 10⁷ → NaN po 1–2 krokach,
  * He: σ = √(2/17) ≈ 0,34 → odchylenie ~1 w każdej warstwie.
* Xavier/Glorot (dla tanh/sigmoidy): σ = √(1/d_in) lub √(2/(d_in + d_out)).

**Dlaczego N(0; 1) uczy się szybko, a generalizuje źle (F1 0,50)?** Ogromne wagi startowe dają ogromne logity – sieć jest od
razu „pewna” losowych odpowiedzi. Koszt treningowy spada (bo α·gradient ~100 to duże kroki), ale duże losowe wagi zostają
i odpowiedzi na nowe dane są skrajne i przypadkowe (minimum kosztu walidacyjnego ~1,9 zamiast ~0,45).

**Martwe neurony:** wielki krok może przesunąć biasy/wagi tak, że wejście neuronu ReLU jest zawsze ujemne → wyjście 0,
gradient 0, neuron „nie żyje”. Gdy umrą wszystkie w jakiejś warstwie, sieć zwraca stałą (σ = 10, trzecie losowanie).

## 2.7 Warunek stopu i przeuczenie

* Szybsza optymalizacja (ADAM, SGD) → szybciej do minimum kosztu treningowego → szybciej przeuczenie.
  Najlepsza epoka na walidacji: GD pełny zbiór ~464, paczki ~83, SGD ~4, ADAM α = 0,01 ~10.
* Dlatego early stopping (pamiętanie wag z najniższego kosztu walidacyjnego) jest w każdym eksperymencie.
* Przy dobrze dobranym stopie F1 testowe jest podobne (0,76–0,78) – metoda optymalizacji zmienia szybkość
  i to, czy uczenie w ogóle ruszy, a nie najlepszą osiągalną jakość.

# 3. Kod krok po kroku (`optim.py`)

```python
class GD:
    def step(self, pairs):                  # pairs = [(W, dW), (b, db), ...]
        for w, g in pairs:
            w -= self.lr * g                # zmiana w miejscu (tablice NumPy)
```

```python
class Adam:
    def step(self, pairs):
        if self.m is None:                  # pierwsze wywołanie: m = v = 0 dla każdej tablicy wag
            self.m = [np.zeros_like(w) for w, _ in pairs]
            self.v = [np.zeros_like(w) for w, _ in pairs]
        self.t += 1                         # numer kroku do korekcji biasu
        for (w, g), m, v in zip(pairs, self.m, self.v):
            m *= self.beta1; m += (1 - self.beta1) * g
            v *= self.beta2; v += (1 - self.beta2) * g * g
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            w -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

* `MLP.params_and_grads()` zwraca pary (parametr, gradient) wszystkich warstw liniowych – optymalizator nie musi
  znać struktury sieci,
* `Linear(..., init="he")` – σ = √(2/n_in), inaczej `init="normal", scale=σ`.

```python
def train(model, opt, X, y, Xv, yv, mode="full", batch=32, epochs=500, seed=SEED):
    bs = {"full": n, "minibatch": batch, "single": 1}[mode]
    for ep in range(1, epochs + 1):
        order = rng.permutation(n) if bs < n else np.arange(n)   # tasowanie przed epoką
        for s in range(0, n, bs):
            idx = order[s:s + bs]
            model.loss_and_grad(X[idx], y[idx])                    # forward + backward na paczce
            opt.step(model.params_and_grads())                     # jeden krok optymalizatora
        log(ep)                                                    # koszt na CAŁYM zbiorze train/val
```

* koszt w `SigmoidBCE` jest średnią po paczce, więc gradient ma tę samą skalę niezależnie od B,
* `history` ma `epoch`, `step` (liczba kroków), `time`, `loss`, `val_loss`, `grad_norms` – można rysować w funkcji
  epok, kroków albo czasu,
* `best_params`, `best_epoch` – early stopping; przy NaN uczenie jest przerywane.

`main()`: siodło 2D (GD vs ADAM, α = 0,1), tryby (He, GD α = 0,01, 1000 epok), inicjalizacja (paczki, GD α = 0,01,
500 epok, N(0; 0,01) / N(0; 1) / He / N(0; 10)), GD vs ADAM (pełny zbiór, 3000 epok, start He i N(0; 0,01)).
Wszystko na 3 losowaniach.

`test_optim.py`: ADAM vs wzory, pierwszy krok = α, GD, liczba kroków w trybach, gradient pełny = średnia pojedynczych,
wariancja He vs N(0; 1), propagacja wsteczna vs numeryczna.

# 4. Wyniki w pigułce – jak je czytać

| eksperyment | wynik | wyjaśnienie |
|---|---|---|
| siodło 2D | GD 91 kroków, ADAM 6 | ADAM normalizuje krok, mały gradient ≠ mały krok |
| tryby, epoki do kosztu 0,3 | 398 / 66 / 4 | 1 / 6 / 181 kroków na epokę |
| tryby, kroki do kosztu 0,3 | 398 / 398 / 724 | postęp na krok podobny, SGD zaszumiony |
| tryby, czas 1000 epok | 0,3 / 0,8 / ~15 s | małe operacje są nieefektywne |
| N(0; 0,01) | stoi na 0,69 | zanikający sygnał i gradient (~10⁻⁸) |
| N(0; 1) | koszt start 12, F1 0,50 | eksplozja, duże losowe wagi zostają |
| He | F1 0,77, stabilnie | wariancja zachowana |
| N(0; 10) | NaN / martwa sieć | gradient ~10⁷ |
| GD vs ADAM, start He | 398 vs 10–63 epok do kosztu 0,3 | momentum + skalowanie |
| GD vs ADAM, start N(0; 0,01) | GD stoi 3000 epok, ADAM rusza po 17–90 | krok ~α niezależnie od skali gradientu |

# 5. Pytania na wejściówkę (z odpowiedziami)

1. **Dlaczego uczenie głębokich sieci jest problemem niewypukłym?** Złożenie nieliniowych warstw; np. symetria permutacji neuronów daje wiele równoważnych minimów.
2. **Co jest głównym problemem optymalizacji sieci głębokich?** Punkty siodłowe i płaskowyże (gradient ≈ 0), a nie minima lokalne.
3. **Co to jest punkt siodłowy?** Punkt z zerowym gradientem, w którym funkcja w jednych kierunkach rośnie, w innych maleje (hesjan nieokreślony).
4. **Dlaczego prosty GD radzi sobie źle w punkcie siodłowym?** Krok = α·g, a g ≈ 0 – postęp tym wolniejszy, im bliżej siodła.
5. **Czym różni się batch, mini-batch i SGD?** Gradient z całego zbioru / z paczki / z jednego przykładu; 1 / ⌈N/B⌉ / N kroków na epokę.
6. **Co to jest epoka, a co iteracja?** Epoka – przejście przez cały zbiór; iteracja – jedna aktualizacja wag.
7. **Dlaczego SGD może pomóc wyjść z punktu siodłowego?** Gradient jest zaszumiony (przybliżenie kosztu), więc szum wytrąca z płaskiego miejsca.
8. **Zalety mini-batch?** Szum pomagający w optymalizacji, wydajne mnożenie macierzy, mniejsza pamięć niż pełny zbiór.
9. **Po co tasować dane co epokę?** Żeby paczki były różne i gradient był nieobciążonym, losowym estymatorem.
10. **Na czym polega momentum?** Krok zachowuje kierunek poprzednich kroków (średnia krocząca gradientów) – przejście przez siodło „z rozpędu”.
11. **Na czym polega skalowanie przez odwrotność wielkości gradientu?** Dzielimy krok przez √(średnia g²) – na płaskich obszarach krok rośnie.
12. **Zapisz wzory ADAM.** m = β₁m + (1−β₁)g; v = β₂v + (1−β₂)g²; m̂ = m/(1−β₁ᵗ); v̂ = v/(1−β₂ᵗ); w ← w − α m̂/(√v̂ + ε).
13. **Po co korekcja biasu w ADAM?** m i v startują od zera, więc na początku są zaniżone; korekcja to wyrównuje (pierwszy krok = α·znak(g)).
14. **Po co ε?** Zabezpieczenie przed dzieleniem przez zero.
15. **Typowe wartości hiperparametrów ADAM?** α = 0,001, β₁ = 0,9, β₂ = 0,999, ε = 10⁻⁸.
16. **Dlaczego ADAM poradził sobie z inicjalizacją N(0; 0,01), a GD nie?** Krok ADAM ma długość ~α niezależnie od skali gradientu; gradient 10⁻⁸ nie spowalnia ADAM.
17. **Czy ADAM daje lepszy model?** Nie w naszych wynikach – szybciej osiąga ten sam F1 (0,76–0,77), ale szybciej się przeucza.
18. **Co to są zanikające gradienty i skąd się biorą?** Wielokrotne mnożenie przez macierze wag zmniejszające wariancję → gradient maleje wykładniczo w stronę wejścia.
19. **Co to są eksplodujące gradienty?** Odwrotnie: wagi zwiększające wariancję → gradient i aktywacje rosną wykładniczo (NaN).
20. **Wyprowadź inicjalizację He.** Var(y) = d_in·Var(W)·E[x²], po ReLU E[h²] = Var(z)/2 → Var(W) = 2/d_in.
21. **Dlaczego N(0; 1) jest złą inicjalizacją dla tej sieci?** d_in·σ²/2 = 8,5 > 1 – sygnał rośnie ok. 3× na warstwę, koszt startowy 12, słaba generalizacja.
22. **Co się stało przy N(0; 10)?** Gradient ~10⁷ → pierwszy krok wyrzucił wagi daleko → NaN albo wszystkie ReLU martwe.
23. **Co to jest martwy neuron ReLU?** Neuron z zawsze ujemnym wejściem – wyjście i gradient zerowe, nie wraca do życia.
24. **Czemu nie inicjalizować zerami?** Symetria – neurony warstwy dostają ten sam gradient i zostają identyczne.
25. **Jak dobrałeś głębokość i warunek stopu?** 5 warstw ukrytych – płytsza sieć ukrywa problemy (z He jedna warstwa wystarcza); stała liczba epok, żeby było widać, kto nie ruszył, plus early stopping do oceny jakości.
26. **Dlaczego porównujesz tryby także w krokach i czasie?** W epokach SGD wygrywa tylko dlatego, że robi 181 kroków na epokę; uczciwe porównanie to postęp na krok i na sekundę.

# 6. Modyfikacje, o które może poprosić prowadzący (gotowe fragmenty)

**Momentum (bez skalowania)**:
```python
class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None
    def step(self, pairs):
        if self.v is None:
            self.v = [np.zeros_like(w) for w, _ in pairs]
        for (w, g), v in zip(pairs, self.v):
            v *= self.beta; v += g
            w -= self.lr * v
```

**RMSProp (samo skalowanie)**: `v = b2*v + (1-b2)*g*g; w -= lr * g / (np.sqrt(v) + eps)`.

**ADAM bez korekcji biasu** (do pokazania, po co jest): w `Adam.step` użyj `m` i `v` zamiast `m_hat` i `v_hat`.

**Inny rozmiar paczki**: `train(..., mode="minibatch", batch=8)`.

**Inna głębokość**: `MLP(d, (17,) * 8, "he")`.

**Inicjalizacja Xavier**: w `Linear`: `std = np.sqrt(1.0 / n_in)`.

**Malejący współczynnik uczenia**: w pętli `opt.lr = lr0 / (1 + decay * ep)`.

**Przycinanie gradientu (gradient clipping)** – ratuje przed eksplozją:
```python
norm = np.sqrt(sum(np.sum(g * g) for _, g in pairs))
if norm > 5.0:
    pairs = [(w, g * 5.0 / norm) for w, g in pairs]
```

**Early stopping z przerwaniem**: po aktualizacji `best` w `train`: `if ep - best[2] > patience: break`.

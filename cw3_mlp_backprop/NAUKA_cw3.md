---
title: "Ćwiczenie 3 – materiał do nauki: sieć wielowarstwowa i propagacja wsteczna"
---

Plik do nauki przed wejściówką i obroną. Kolejność: (1) o co chodzi, (2) teoria z wyprowadzeniami,
(3) kod linijka po linijce, (4) jak czytać wyniki eksperymentów, (5) pytania z odpowiedziami,
(6) modyfikacje kodu, o które może poprosić prowadzący.

# 1. O co chodzi w ćwiczeniu

Trzeba **samodzielnie zaimplementować** wielowarstwową sieć neuronową (MLP) uczoną **propagacją wsteczną**:

* przekształcenie liniowe Wx + b, funkcja aktywacji **ReLU**, funkcja kosztu **jak w regresji logistycznej** (sigmoida + entropia krzyżowa),
* każda operacja jako moduł z dwiema funkcjami: `forward` (wynik + zapamiętanie wejścia) i `backward` (gradient po wagach + gradient po wejściu),
* wagi z **N(0; 0,01)**, współczynnik uczenia **α = 0,01**.

Potem zbadać na zbinaryzowanym zbiorze Heart Disease:

* różną **wymiarowość warstwy ukrytej** (mniejszą od liczby cech, równą, dużo większą),
* dane **znormalizowane i nieznormalizowane**,
* różną **liczbę warstw**.

Związek z ćw. 2: sieć bez warstw ukrytych **to jest** regresja logistyczna. Różnica: z warstwą ukrytą funkcja kosztu
**nie jest wypukła**, więc pojawiają się płaskie obszary, punkty siodłowe i zależność od inicjalizacji.

# 2. Teoria

## 2.1 Sieć jako złożenie funkcji

Sieć z jedną warstwą ukrytą: h = ReLU(W₁x + b₁), z = W₂h + b₂, p = σ(z), L = BCE(y, p).

Ogólnie: L = L(f_K(… f₂(f₁(x)))). Każde fₖ to jeden moduł. Sieć z K warstwami liniowymi i ReLU pomiędzy nimi
nazywamy MLP (multilayer perceptron).

**Dlaczego potrzebna jest nieliniowość?** Złożenie przekształceń liniowych jest liniowe: W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂).
Bez ReLU dowolnie głęboka sieć to nadal regresja logistyczna.

## 2.2 Reguła łańcuchowa i propagacja wsteczna

Dla złożenia y = f(x), L = g(y): **∂L/∂x = ∂L/∂y · ∂y/∂x**.

Konwencja z treści ćwiczenia (x – wektor kolumnowy):

* L jest skalarem, więc ∂L/∂x to **wektor wierszowy** o wymiarze xᵀ (żeby zgadzało się mnożenie macierzy),
* ∂y/∂x to **macierz Jacobiego** o wymiarach (dim y, dim x),
* kolejność mnożenia ma znaczenie: ∂L/∂x = (∂L/∂y)(∂y/∂x), czyli (1 × dim y)(dim y × dim x) = (1 × dim x).

Propagacja wsteczna = liczenie gradientów od końca: zaczynamy od ∂L/∂z na wyjściu i mnożymy przez Jacobiany
kolejnych modułów w odwrotnej kolejności. Każdy moduł potrzebuje tylko „nadchodzącego” ∂L/∂y i swoich zapamiętanych
danych z przejścia w przód. Koszt obliczeń: ok. 2–3 razy więcej niż jedno przejście w przód (zamiast osobnego
liczenia gradientu dla każdej wagi).

**Dwie zasady z treści:**

1. Wartości wyjściowe liczymy w kolejności operacji, gradienty w kolejności odwrotnej.
2. Przy przejściu w przód każdy moduł **zapamiętuje swoje wejście**, bo backward go potrzebuje (np. ∂L/∂W = (∂L/∂y)ᵀxᵀ).

## 2.3 Moduł Linear: y = Wx + b – wyprowadzenie

W ma wymiary (n_out, n_in). Dla jednego przykładu, przy danym wektorze wierszowym δ = ∂L/∂y (1 × n_out):

* yᵢ = Σⱼ Wᵢⱼ xⱼ + bᵢ → ∂yᵢ/∂Wᵢⱼ = xⱼ → **∂L/∂Wᵢⱼ = δᵢ xⱼ**, czyli macierzowo **∂L/∂W = δᵀ xᵀ** (iloczyn zewnętrzny, wymiar jak W),
* ∂yᵢ/∂bᵢ = 1 → **∂L/∂b = δ**,
* ∂y/∂x = W → **∂L/∂x = δ W** (1 × n_in) – to idzie dalej do poprzedniego modułu.

**Dla paczki N przykładów** (wiersz macierzy X = jeden przykład, tak jak w kodzie):
Y = X Wᵀ + b, dY = ∂L/∂Y ma kształt (N, n_out):

* `dW = dY.T @ X` (suma iloczynów zewnętrznych po przykładach),
* `db = dY.sum(0)`,
* `dX = dY @ W`.

Średnia po przykładach (1/N) jest już w dY, bo koszt to średnia (patrz 2.5).

## 2.4 Moduł ReLU – mnożenie element-wise zamiast macierzy

ReLU(x) = max(0, x), pochodna: 1 dla x > 0, 0 dla x < 0 (w x = 0 umownie 0).

ReLU działa element-wise, więc Jacobian ∂y/∂x jest **macierzą diagonalną** z [x > 0] na przekątnej.
Mnożenie δ przez macierz diagonalną = mnożenie element-wise: **∂L/∂x = δ ⊙ [x > 0]**.
To uproszczenie z treści (przykład z x² działa tak samo: ∂L/∂x = δ ⊙ 2x).

Zalety ReLU: brak nasycenia dla x > 0 (pochodna 1, gradient nie maleje jak przy sigmoidzie), tanie obliczenia.
Wada: **martwe neurony** – jeśli wejście neuronu jest zawsze ujemne, gradient = 0 i neuron już się nie zmieni.

## 2.5 Wyjście: sigmoida + entropia krzyżowa

p = σ(z), L = −(1/N) Σ [y ln p + (1 − y) ln(1 − p)]. Z ćw. 2: ∂L/∂p · ∂p/∂z = (p − y) / N dla każdego przykładu.

W kodzie sigmoida i BCE są jednym modułem (`SigmoidBCE`), bo:

* pochodna łączna jest prosta: **∂L/∂z = (p − y) / N**,
* osobno ∂L/∂p = (p − y)/(p(1 − p)) przy nasyceniu (p ≈ 0 lub 1) dzieli przez prawie zero – numerycznie niestabilne.

## 2.6 Niewypukłość, płaskowyże i punkty siodłowe

Z warstwą ukrytą koszt nie jest wypukły. Przykład: zamiana miejscami dwóch neuronów ukrytych (z ich wagami) daje
tę samą funkcję, więc minimów jest wiele, a średnia z dwóch minimów zwykle minimum nie jest.

**W = 0 (i jego okolica) to punkt siodłowy/płaskowyż:** wszystkie aktywacje ≈ 0, wyjście z ≈ 0, p ≈ 0,5, a gradient
każdej warstwy zawiera iloczyn wag pozostałych warstw, które są małe → gradient ≈ 0, choć to nie minimum
(koszt 0,69 = koszt stałej odpowiedzi).

## 2.7 Zanikający gradient przy małej inicjalizacji

Gradient pierwszej warstwy to ∂L/∂W₁ = (δ_wyjścia · W_K · D_{K−1} · … · W₂ · D₁)ᵀ xᵀ, gdzie Dₖ = diag([hₖ > 0]).
Każde mnożenie przez macierz o elementach rzędu σ = 0,01 (i ok. 17 składników) zmniejsza wektor kilka do kilkudziesięciu razy.
Z każdą warstwą gradient maleje **wykładniczo**. Wyniki (norma gradientu w epoce 0, warstwa przy wejściu):

| warstw ukrytych | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| ‖∂L/∂W₁‖ | 0,72 | 1,9·10⁻² | 3,1·10⁻⁴ | 9,1·10⁻⁶ | 2,6·10⁻⁷ |
| wyjście z plateau (epoka) | ~22 | ~765 | ~7700 | ~94 600 | nigdy (150 000) |

Ten sam problem przy przejściu w przód: sygnał też maleje z każdą warstwą (aktywacje → 0).

**Lekarstwo: inicjalizacja He** – N(0; √(2/d_in)). Zachowuje wariancję sygnału: Var(Wx) = d_in · σ² · E[x²], a ReLU
zeruje połowę wartości (stąd 2). U nas: 3 i 4 warstwy z He wychodzą z plateau po 24 i 55 epokach.
(Dla sigmoidy/tanh: Xavier/Glorot, √(1/d_in) lub √(2/(d_in + d_out)).)

## 2.8 Dlaczego nie zera? (problem symetrii)

Jeśli wszystkie wagi = 0, to wszystkie neurony w warstwie mają ten sam wkład i dostają **ten sam gradient**, więc
na zawsze pozostają identyczne (sieć zachowuje się jak jeden neuron na warstwę). Dodatkowo przy ReLU i W = 0
gradient warstw ukrytych jest dokładnie zerowy. Dlatego wagi losujemy (łamanie symetrii), a biasy mogą być zerami.
W regresji logistycznej (ćw. 2) zera były w porządku – jeden neuron, nie ma czego łamać.

## 2.9 Normalizacja danych

* **Standaryzacja**: (x − μ)/σ, statystyki z treningu. **Min-max**: (x − min)/(max − min) do [0, 1].
* Bez normalizacji cechy o dużych wartościach (cholesterol ~250) dają ogromne składowe gradientu → przy α = 0,01
  krok jest dla nich za duży (oscylacje, rozbieżność), a dla cech 0/1 za mały. Wskaźnik uwarunkowania XᵀX: surowe
  5,5·10⁶, min-max 250, standaryzacja 46 (ćw. 2: próg stabilności α < 2/λ_max).
* U nas: regresja logistyczna na surowych danych się rozbiega (koszt ~8), MLP utyka na 0,52 (F1 0,67).
  MLP nie wybucha, bo wagi są małe, a ReLU część sygnału zeruje, ale też się nie uczy.
* Min-max działa, ale wolniej niż standaryzacja (cechy nie mają średniej 0, a zera po min-max „wyłączają” wagi).

## 2.10 Szerokość warstwy ukrytej

* liczba parametrów: (d + 1)·h + (h + 1) – dla h = 2: 39, h = 17: 324, h = 256: 4865 przy 181 przykładach treningowych,
* **szersza** warstwa: krótsze plateau (gradient wag wyjściowych to suma po większej liczbie neuronów), większa pojemność,
  szybsze i silniejsze przeuczenie; **węższa**: może nie mieć pojemności (h = 2 nie schodzi poniżej kosztu 0,18),
* teoria: twierdzenie o uniwersalnej aproksymacji – jedna warstwa ukryta z odpowiednio wieloma neuronami może
  przybliżyć dowolną funkcję ciągłą. Nie mówi nic o tym, czy uczenie ją znajdzie i czy będzie dobrze generalizować.

## 2.11 Przeuczenie i early stopping

Sieć z warstwą ukrytą może zapamiętać zbiór treningowy (koszt 0,03, accuracy 98%), a koszt walidacyjny rośnie
(0,39 → 0,91). Remedium: **early stopping** (wagi z epoki z najniższym kosztem walidacyjnym), regularyzacja L2,
mniejsza sieć, więcej danych. F1 może się prawie nie zmieniać, bo zależy tylko od strony progu 0,5.

# 3. Kod krok po kroku (`mlp.py`)

```python
class Linear:
    def forward(self, X):
        self.X = X                         # zapamiętane wejście
        return X @ self.W.T + self.b       # Y = X W^T + b  (wiersz = przykład)
    def backward(self, dY):                # dY = dL/dY, (N, n_out)
        self.dW = dY.T @ self.X            # dL/dW (n_out, n_in)
        self.db = dY.sum(0)                # dL/db
        return dY @ self.W                 # dL/dX -> do poprzedniego modułu
```

```python
class ReLU:
    def forward(self, X):
        self.mask = X > 0                  # zapamiętana maska = pochodna
        return X * self.mask
    def backward(self, dY):
        return dY * self.mask              # Jacobian diagonalny -> mnożenie element-wise
```

```python
class SigmoidBCE:
    def forward(self, z, y):
        self.p, self.y = sigmoid(z), y
        return bce(y, self.p)
    def backward(self):
        return ((self.p - self.y) / len(self.y))[:, None]   # dL/dz, kształt (N, 1)
```

Klasa `MLP`:

* `__init__(n_in, hidden=(17,), init="normal", init_scale=0.01)` – buduje listę `Linear, ReLU, …, Linear`;
  `hidden=()` → regresja logistyczna; `init="he"` → wagi He,
* `forward` – pętla po modułach w przód, zwraca logit z,
* `loss_and_grad(X, y)` – koszt + **pętla po modułach wstecz** (`reversed(self.layers)`), każdy zapisuje swoje dW, db,
* `step(lr)` – w ← w − α·dW dla każdej warstwy liniowej,
* `fit(...)` – pełny batch GD; co epokę koszt train/val, co `log_every` epok F1/P/R i normy gradientu;
  pamięta wagi z najniższym kosztem walidacyjnym (`best_epoch`, `best_params`),
* `epochs_to(loss, 0.6)` – „wyjście z plateau”: pierwsza epoka z kosztem < 0,6 (koszt stałej odpowiedzi to 0,69).

`main()` – eksperymenty: model bazowy 17-17-1 (30 000 epok), szerokości 2/4/17/68/256 (3 losowania), skalowanie
surowe/min-max/standaryzacja (MLP i regresja logistyczna), liczba warstw 0–4 (150 000 epok, 3 losowania),
dodatkowo He dla 3 i 4 warstw.

`test_mlp.py` – gradient każdego modułu i całej sieci vs różnice skończone, 0 warstw = wzór z ćw. 2, spadek kosztu.

# 4. Wyniki w pigułce – jak je czytać

| eksperyment | co się dzieje | dlaczego |
|---|---|---|
| 17-17-1, N(0; 0,01) | plateau ~720 epok na koszcie 0,69, potem uczenie; najlepsza walidacja w epoce 1783, F1 test 0,764 | małe wagi → p ≈ 0,5, gradient ≈ 0 |
| długi trening | koszt train 0,03, val 0,39 → 0,91 | przeuczenie, sieć ma 324 parametry na 181 przykładów |
| szerokość 2 → 256 | plateau 1055 → 428 epok, F1 przy early stopping ~0,76 dla wszystkich | szerokość zmienia szybkość i pojemność, nie jakość |
| dane surowe | regresja log. rozbiega się, MLP utyka (0,52, F1 0,67) | złe uwarunkowanie, α za duże dla dużych cech |
| min-max | działa, wolniej niż standaryzacja | gorsze uwarunkowanie (250 vs 46) |
| 0–4 warstwy | plateau ×10 na każdą warstwę; 4 warstwy stoją | zanikający gradient |
| He | 3–4 warstwy uczą się od razu | wariancja sygnału zachowana |

# 5. Pytania na wejściówkę (z odpowiedziami)

1. **Na czym polega propagacja wsteczna?** Na liczeniu gradientu kosztu po wszystkich wagach regułą łańcuchową, od wyjścia do wejścia, z użyciem wartości zapamiętanych w przejściu w przód.
2. **Zapisz regułę łańcuchową.** ∂L/∂x = ∂L/∂y · ∂y/∂x.
3. **Jaki wymiar ma ∂L/∂x, a jaki ∂y/∂x?** ∂L/∂x – wektor wierszowy (1 × dim x); ∂y/∂x – macierz (dim y × dim x).
4. **Co robi forward, a co backward modułu?** Forward liczy wyjście i zapamiętuje wejście; backward dostaje ∂L/∂y, liczy gradient wag i zwraca ∂L/∂x.
5. **Dlaczego forward musi zapamiętać wejście?** Bo pochodne zależą od wejścia, np. ∂L/∂W = δᵀxᵀ, maska ReLU [x > 0].
6. **W jakiej kolejności liczymy forward i backward?** Forward w kolejności operacji, backward w odwrotnej.
7. **Wyprowadź gradienty warstwy liniowej.** ∂L/∂W = δᵀxᵀ, ∂L/∂b = δ, ∂L/∂x = δW (δ = ∂L/∂y).
8. **Pochodna ReLU i dlaczego nie mnożymy przez macierz?** [x > 0]; Jacobian jest diagonalny, więc mnożenie = mnożenie element-wise.
9. **Ile wynosi ∂L/∂z dla sigmoidy z BCE?** (p − y)/N.
10. **Po co nieliniowa funkcja aktywacji?** Bez niej złożenie warstw liniowych jest liniowe – sieć = regresja logistyczna.
11. **Czym sieć bez warstw ukrytych różni się od regresji logistycznej?** Niczym – to ta sama funkcja i ten sam gradient (sprawdzone w testach).
12. **Czy koszt MLP jest wypukły?** Nie (np. symetria permutacji neuronów); możliwe płaskowyże, punkty siodłowe, wiele minimów.
13. **Dlaczego sieć z N(0; 0,01) długo stoi na koszcie 0,69?** Małe wagi → p ≈ 0,5 dla wszystkich, gradient jest iloczynem małych wag → płaskowyż/punkt siodłowy, nie minimum.
14. **Co to jest zanikający gradient?** Gradient maleje wykładniczo w stronę wejścia, bo jest mnożony przez kolejne macierze wag; pierwsze warstwy prawie się nie uczą.
15. **Jak liczba warstw wpływa na plateau w twoich wynikach?** Ok. 10 razy dłużej na każdą warstwę; 4 warstwy nie ruszyły w 150 000 epok.
16. **Czemu nie inicjalizować zerami?** Symetria – wszystkie neurony w warstwie dostają ten sam gradient i zostają identyczne; przy ReLU gradient wręcz = 0.
17. **Co to jest inicjalizacja He?** N(0; √(2/d_in)) – zachowuje wariancję sygnału przy ReLU (połowa wartości zerowana → czynnik 2).
18. **Jak szerokość warstwy ukrytej wpływa na uczenie?** Szersza: krótsze plateau, większa pojemność, szybsze przeuczenie; przy early stopping F1 podobne.
19. **Dlaczego normalizacja jest potrzebna?** Cechy o różnych skalach → złe uwarunkowanie; jedno α za duże dla jednych i za małe dla innych kierunków.
20. **Standaryzacja vs min-max?** Standaryzacja: średnia 0, odchylenie 1; min-max: [0, 1]. Statystyki zawsze z treningu. U nas standaryzacja szybsza.
21. **Co to jest martwy neuron ReLU?** Neuron z zawsze ujemnym wejściem: wyjście 0, gradient 0, nie wraca do życia.
22. **Jak sprawdzić poprawność backward?** Gradient numeryczny (L(w + h) − L(w − h))/2h dla każdej wagi; u nas zgodność do 10⁻¹⁰.
23. **Czy więcej parametrów to lepszy model?** Nie – na 181 przykładach każda sieć przy early stopping ma F1 ~0,76, tak jak regresja logistyczna; większe sieci szybciej się przeuczają.
24. **Dlaczego F1 prawie się nie zmienia, gdy koszt walidacyjny rośnie?** F1 zależy tylko od progu 0,5; koszt od pewności predykcji.
25. **Co to jest epoka w tym ćwiczeniu?** Jedno przejście przez cały zbiór = jeden krok (pełny batch).
26. **Ile kosztuje backward w porównaniu z forward?** Ten sam rząd (ok. 2× więcej mnożeń macierzy), zamiast osobnego przejścia dla każdej wagi.

# 6. Modyfikacje, o które może poprosić prowadzący (gotowe fragmenty)

**Inna funkcja aktywacji (tanh)**:
```python
class Tanh:
    def forward(self, X):
        self.y = np.tanh(X); return self.y
    def backward(self, dY):
        return dY * (1 - self.y ** 2)
```

**Sigmoida jako aktywacja ukryta**: `self.s = sigmoid(X)`, backward: `dY * self.s * (1 - self.s)`.

**Leaky ReLU** (brak martwych neuronów): forward `np.where(X > 0, X, 0.01 * X)`, backward `dY * np.where(X > 0, 1, 0.01)`.

**Inicjalizacja He / Xavier**: `MLP(d, (17, 17), init="he")`; Xavier: `std = np.sqrt(1.0 / n_in)`.

**Regularyzacja L2** (w `step` albo po `backward`): `layer.dW += lam * layer.W` i do kosztu `0.5 * lam * sum(W**2)`.

**Mini-batch**: w `fit` zamiast jednego `loss_and_grad(X, y)` pętla po paczkach `idx = perm[s:s+32]`.

**Moduł kwadratu z treści**:
```python
class Square:
    def forward(self, x):
        self.x = x; return x * x
    def backward(self, dY):
        return dY * 2 * self.x
```

**Dowolna architektura**: `MLP(d, (64, 32, 8))` – trzy warstwy ukryte o różnej szerokości.

**Early stopping z przerwaniem uczenia**: w `fit` po aktualizacji `best`: `if ep - best[2] > patience: break`, potem `set_params(best_params)`.

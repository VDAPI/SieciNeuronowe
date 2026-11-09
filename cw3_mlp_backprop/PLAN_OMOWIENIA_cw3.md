# Plan omówienia raportu – ćw. 3

Krótko, po 2–4 zdania na punkt. W nawiasie: co pokazać palcem.

## Cel i implementacja (rozdz. 1, 2.1–2.3)
- „Zaimplementowałem sieć wielowarstwową uczoną propagacją wsteczną: każda operacja to moduł z forward i backward.”
- „Forward liczy wynik i zapamiętuje wejście, backward dostaje dL/dy, liczy gradient wag i zwraca dL/dx dla poprzedniego modułu. W przód w kolejności operacji, wstecz w odwrotnej.”
- (tabela w 2.1) „Linear: dL/dW = δᵀxᵀ, dL/db = δ, dL/dx = δW. ReLU: mnożę przez maskę x > 0, bo Jacobian jest diagonalny. Na wyjściu sigmoida z entropią krzyżową, pochodna (p − y)/N, jak w ćw. 2.”
- „Wagi z N(0; 0,01), α = 0,01, pełny batch, dane z ćw. 1: 17 cech, podział 181 / 61 / 61.”

## Testy (2.4, tabela)
- „Gradient z backward porównałem z numerycznym dla każdej wagi sieci z dwiema warstwami ukrytymi – zgodność do 10⁻¹⁰.”
- „Sieć bez warstw ukrytych daje dokładnie gradient regresji logistycznej z ćw. 2.”

## Model bazowy (rozdz. 3, Rys. 1)
- (Rys. 1) „Przez ok. 720 epok koszt stoi na 0,69, F1 = 0: wagi są tak małe, że sieć mówi p ≈ 0,5 każdemu. To płaskowyż, a nie minimum.”
- „Potem uczy się szybko, najlepsza walidacja w epoce 1783 – F1 test 0,764.”
- „Dalej się przeucza: koszt treningowy 0,03, walidacyjny rośnie do 0,91. Dlatego zapamiętuję wagi z najlepszej epoki (early stopping).”

## Szerokość warstwy ukrytej (rozdz. 4, Rys. 2)
- „2 i 4 neurony (mniej niż cech), 17 (równo), 68 i 256 (dużo więcej).”
- „Szersza sieć szybciej wychodzi z plateau (1055 → 428 epok) i szybciej się przeucza. Wąska (2) nie ma pojemności, koszt zostaje na 0,18.”
- „Przy early stopping F1 ~0,76 dla każdej szerokości – szerokość zmienia szybkość, nie jakość.”

## Normalizacja (rozdz. 5, Rys. 3)
- „Na surowych danych uwarunkowanie 5,5 mln: regresja logistyczna się rozbiega, sieć utyka na 0,52, F1 0,67.”
- „Min-max i standaryzacja działają, standaryzacja szybciej (uwarunkowanie 46 vs 250).”

## Liczba warstw (rozdz. 6, Rys. 4)
- (Rys. 4 lewy) „Każda warstwa wydłuża plateau ok. 10 razy: 22 / 765 / 7700 / 94 600 epok. Cztery warstwy nie ruszyły w 150 000 epok.”
- (Rys. 4 prawy) „Powód: gradient maleje kilka do kilkudziesięciu razy na każdą warstwę w stronę wejścia – przy 4 warstwach pierwsza ma gradient 3·10⁻⁷. To zanikający gradient.”
- „Głębsze sieci, które ruszyły, nie są lepsze – F1 ~0,75–0,76.”

## Dodatkowo He (rozdz. 7, Rys. 5)
- „Z inicjalizacją He te same 3 i 4 warstwy uczą się od razu (24 i 55 epok). Winna jest za mała inicjalizacja, a nie głębokość.”

## Wnioski (rozdz. 8–9)
- Przeczytaj tabelę z rozdz. 8. Puenta: „Implementacja działa, ale przy N(0; 0,01) głębsza sieć startuje z płaskowyżu, z którego zwykły gradient wychodzi wykładniczo dłużej z każdą warstwą. Normalizacja jest konieczna, szerokość zmienia głównie szybkość i przeuczenie.”

## Możliwe pytania prowadzącego
- **Dlaczego backward ReLU to mnożenie element-wise?** – Jacobian funkcji element-wise jest diagonalny, mnożenie przez niego = mnożenie przez przekątną.
- **Skąd dL/dW = δᵀxᵀ?** – yᵢ = Σⱼ Wᵢⱼxⱼ + bᵢ, więc ∂L/∂Wᵢⱼ = δᵢxⱼ – iloczyn zewnętrzny.
- **Czemu sigmoida i BCE razem?** – Pochodna łączna (p − y)/N jest prosta i stabilna, osobno dzieliłbym przez p(1 − p) ≈ 0.
- **Czy to plateau to minimum lokalne?** – Nie: koszt 0,69 to koszt stałej odpowiedzi, a He od razu z niego schodzi; gradient jest mały, bo wagi są małe.
- **Dlaczego nie zera?** – Symetria: wszystkie neurony dostają ten sam gradient i zostają identyczne; z ReLU gradient wręcz zerowy.
- **Dlaczego surowa regresja się rozbiega, a sieć nie?** – Regresja ma krok α·cholesterol, a próg stabilności to ok. 7·10⁻⁵ (ćw. 2); w sieci małe wagi i ReLU tłumią sygnał, ale uczenie i tak stoi.
- **Co zmienić w kodzie, żeby…** – warstwy: `MLP(d, (17, 17))`; szerokość: `MLP(d, (256,))`; inicjalizacja: `init="he"`; dane: `load_data("minmax")`; α: `fit(..., lr=...)`.

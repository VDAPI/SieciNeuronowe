# Plan omówienia raportu – ćw. 3, cz. 2

Krótko, po 2–4 zdania na punkt. W nawiasie: co pokazać palcem.

## Cel i implementacja (rozdz. 1–2)
- „W sieci z dwiema i więcej warstwami koszt nie jest wypukły, ale głównym problemem nie są minima lokalne, tylko punkty siodłowe i płaskowyże, gdzie gradient jest prawie zerowy.”
- „Do sieci z cz. 1 dopisałem trzy tryby uczenia (pełny zbiór, paczki po 32, pojedyncze przykłady), optymalizator ADAM i wybór inicjalizacji.”
- (tabela 2.1) „ADAM trzyma dla każdej wagi średnią gradientu m – to momentum – i średnią kwadratu v – to skalowanie kroku. Korekcja biasu, bo obie startują od zera.”
- „Sieć ma 5 warstw ukrytych po 17 neuronów – celowo za głęboka, żeby problemy było widać.”

## Testy (2.4)
- „ADAM zgadza się ze wzorami, pierwszy krok ma długość α niezależnie od skali gradientu, gradient pełnego zbioru to średnia pojedynczych, backward zgodny z numerycznym do 10⁻¹⁰.”

## Punkt siodłowy (rozdz. 3, Rys. 1)
- (Rys. 1) „Funkcja z siodłem w (0, 0). GD zjeżdża do siodła i stoi przy nim, bo gradient w kierunku y jest prawie zerowy – ucieka po 91 krokach. ADAM po 6, bo dzieli krok przez wielkość gradientu.”

## Tryby uczenia (rozdz. 4, Rys. 2)
- (Rys. 2 lewy) „W epokach: pojedyncze przykłady 4, paczki 66, pełny zbiór 398 epok do kosztu 0,3.”
- (Rys. 2 środkowy) „Ale w krokach wszystkie potrzebują podobnie: 398–724. SGD jest najbardziej zaszumiony.”
- „W czasie paczki są najszybsze, 1000 epok SGD trwa ok. 15 s zamiast 0,3 s.”
- (Rys. 2 prawy) „Szybsze tryby szybciej się przeuczają – najlepsza epoka na walidacji: 464 / 83 / 4. F1 podobne, 0,76–0,78.”

## Inicjalizacja (rozdz. 5, Rys. 3)
- (Rys. 3 środkowy) „N(0; 0,01): sygnał maleje ok. 30× na warstwę – sieć stoi. N(0; 1): rośnie ok. 3× – koszt startowy 12. N(0; 10): ok. 30× – NaN po 1–2 epokach albo wszystkie ReLU martwe. He: odchylenie ok. 1 w każdej warstwie.”
- „N(0; 1) obniża koszt treningowy, ale F1 test tylko 0,50 – duże losowe wagi zostają. He: F1 0,77.”

## GD vs ADAM (rozdz. 6, Rys. 4)
- (Rys. 4 lewy) „Start He: ADAM dochodzi do kosztu 0,3 po 10–63 epokach, GD po 398.”
- (Rys. 4 prawy) „Start N(0; 0,01): GD nie ruszył przez 3000 epok, ADAM po 17–90. Dla ADAM gradient 10⁻⁸ jest tak samo dobry jak 1.”
- „Jakość podobna (F1 ~0,77), ADAM tylko szybciej – i szybciej się przeucza, więc early stopping jest konieczny.”

## Wnioski (rozdz. 7–8)
- Przeczytaj tabelę z rozdz. 7. Puenta: „Problemem były płaskie obszary, nie minima lokalne. Inicjalizacja decyduje, czy uczenie ruszy (He), ADAM z płaskowyżu wychodzi, a tryb uczenia to kompromis między szumem, liczbą kroków i czasem.”

## Możliwe pytania prowadzącego
- **Po co korekcja biasu?** – m i v zaczynają od zera, bez korekcji pierwsze kroki są źle przeskalowane; po korekcji pierwszy krok to α·znak(g).
- **Dlaczego He to √(2/d_in)?** – Var(Wx) = d_in·σ²·E[x²], ReLU zeruje połowę, więc 2/d_in utrzymuje wariancję.
- **Czemu SGD jest najszybszy w epokach, a najwolniejszy w czasie?** – 181 kroków na epokę, ale każdy to małe, nieefektywne mnożenie macierzy.
- **Czy ADAM zawsze lepszy?** – Szybszy, ale nie lepszy na teście; szybciej się przeucza, za duże α też potrafi zepsuć sieć.
- **Skąd NaN przy N(0; 10)?** – Aktywacje ~10⁸, gradient ~10⁷, krok α·10⁷ wyrzuca wagi do nieskończoności.
- **Jak dobrałeś głębokość?** – 5 warstw: przy jednej warstwie i He wszystkie metody działają prawie tak samo, problemy by się schowały.
- **Co zmienić w kodzie, żeby…** – tryb: `train(..., mode="single")`; paczka: `batch=8`; optymalizator: `Adam(0.001)` / `GD(0.01)`; inicjalizacja: `MLP(d, NET, "normal", 1.0)` albo `"he"`.

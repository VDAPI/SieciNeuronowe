# Plan omówienia raportu – ćw. 2

Krótko, po 2–4 zdania na punkt. W nawiasie: co pokazać palcem.

## Str. 2 – Cel i model (rozdz. 1, 2.1–2.3)
- „Zaimplementowałem sam regresję logistyczną uczoną spadkiem gradientu – to najprostsza, jednowarstwowa sieć.”
- „Model: p = σ(Wx + b), sigmoida daje liczbę 0–1, czyli prawdopodobieństwo choroby. Klasa 1, gdy p ≥ 0,5.”
- „Koszt: entropia krzyżowa. Gradient to −(1/N)·Xᵀ(y − p), krok: w ← w − α·gradient, na wszystkich próbkach naraz (pełny batch).”
- „Koszt jest wypukły – ma jedno minimum i nie ma lokalnych. To jest klucz do całego ćwiczenia.”
- „Dane i preprocessing z ćw. 1: 17 cech, podział 181 / 61 / 61.”

## Str. 2–3 – Minimum i testy (2.4, Tabela 1)
- „Minimum L* = 0,287 wyznaczyłem bardzo długim treningiem. Zbieżność = koszt ≤ L* + 0,001.”
- (Tabela 1) „Sprawdziłem gradient numerycznie (różnica 10⁻¹⁰), metryki i minimum zgadzają się ze sklearn.”

## Str. 3–4 – Model bazowy (rozdz. 3, Tabela 2, Rys. 1–2)
- (Tabela 2) „α = 0,1, wagi zerowe, 2000 epok: F1 test 0,786, zbiegł po 1067 epokach.”
- (Rys. 1) „Koszt treningowy spada cały czas, walidacyjny w pewnym momencie zaczyna rosnąć – wrócę do tego przy warunku stopu.”
- (Rys. 2) „Największe wagi: cp_4, thalach (ujemna), sex, ca, thal_7 – te same cechy, co w korelacjach z ćw. 1.”

## Str. 4–5 – Współczynnik uczenia (rozdz. 4, Tabela 3, Rys. 3–4)
- „Trzy strefy:”
  - „α ≤ 0,01 – za wolno, po 5000 epok nie dochodzi do minimum,”
  - „α od 0,1 do 5 – dochodzi do tego samego L*; większe α = mniej epok (1067 → 106 → 18),”
  - „α ≥ 20 – krok przeskakuje minimum, koszt skacze w kółko (Rys. 4).”
- „Lepsze F1 przy α = 20 to przypadek – jedna próbka na 61 zmienia F1 o 0,02.”

## Str. 5–6 – Inicjalizacja (rozdz. 5, Tabela 4, Rys. 5)
- „Małe wagi startowe (zera, N(0; 1), U(−1; 1)) – wszystkie trafiają w to samo minimum.”
- „Duże (N(0; 10), stała 5) – sigmoida się nasyca (p ≈ 0 lub 1), gradient jest mały, uczenie trwa dużo dłużej.”
- „Ale ‖w − w*‖ cały czas maleje – więc to tylko wydłuża uczenie, wynik byłby ten sam.”

## Str. 6–7 – Warunek stopu (rozdz. 6, Tabela 5, Rys. 6)
- (Rys. 6) „Koszt walidacyjny jest najniższy ok. 133. epoki, potem rośnie – model zaczyna się przeuczać.”
- „Early stopping i stop przy |ΔL| < 10⁻⁴ kończą po ~150 epokach z tym samym F1 co 20 000 epok.”
- „F1 prawie się nie zmienia, bo zależy tylko od strony progu 0,5, a koszt też od pewności.”

## Str. 7 – Brak standaryzacji (rozdz. 7, Tabela 6, Rys. 7)
- „Na surowych danych minimum jest to samo, ale uwarunkowanie rośnie z 46 do 5,5 mln.”
- „Nie ma dobrego α: małe jest za wolne dla płaskich kierunków, większe oscyluje na stromym (chol). F1 spada do ~0,65.”

## Str. 8 – Wnioski
- Przeczytaj wnioski. Puenta: „Wypukłość gwarantuje jedno minimum, ale nie gwarantuje, że do niego dojdziemy – złe α i brak standaryzacji to blokują, zła inicjalizacja tylko wydłuża.”

## Możliwe pytania prowadzącego
- **Czy klasyfikator liniowy zawsze zbiega do globalnego minimum?** – Jeśli minimum istnieje i α nie jest za duże (u nas do ok. 5), to tak. Nie: przy za dużym α (oscylacje), przy braku standaryzacji (w praktyce za wolno) i teoretycznie przy danych liniowo separowalnych (minimum w ∞, u nas nie występuje).
- **Skąd gradient (y − p)·x?** – Pochodna sigmoidy σ' = σ(1 − σ) skraca się z pochodną logarytmu w entropii krzyżowej.
- **Czemu nie MSE?** – Z sigmoidą MSE nie jest wypukłe i ma mały gradient przy nasyceniu; entropia krzyżowa jest wypukła.
- **Po co walidacja, skoro koszt jest wypukły?** – Wypukłość dotyczy kosztu treningowego; na nowych danych model i tak może się przeuczyć (rys. 6).
- **Dlaczego duże wagi startowe spowalniają?** – σ nasycona → σ' ≈ 0 dla dobrze sklasyfikowanych, małe kroki.
- **Co zmienić w kodzie, żeby…** – α: `lr=` w `LogisticRegressionGD(...)`; inicjalizacja: `init="normal", init_scale=…`; stop: `stop="early", patience=…`.

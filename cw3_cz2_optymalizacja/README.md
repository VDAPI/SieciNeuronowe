# Ćwiczenie 3, cz. 2 – optymalizacja sieci: tryby uczenia, inicjalizacja, ADAM (własna implementacja, NumPy)

- `raport_cw3_cz2.ipynb` – raport jako notatnik Jupyter (opis + kod + wykresy, już uruchomiony)
- `raport_cw3_cz2.docx` – ten sam raport bez kodu (do oddania)
- `PLAN_OMOWIENIA_cw3_cz2.md` – co mówić przy każdym rozdziale raportu + pytania prowadzącego
- `NAUKA_cw3_cz2.md/.docx` – teoria z wyprowadzeniami, objaśnienie kodu, pytania na wejściówkę
- `optim.py` – sieć z cz. 1, optymalizatory GD i ADAM, uczenie w trybach pełny zbiór / paczki / pojedyncze przykłady,
  inicjalizacja N(0; σ) i He, wszystkie eksperymenty
- `data.py` – preprocessing (taki sam jak w ćw. 1–3)
- `test_optim.py` – testy poprawności (ADAM vs wzory, korekcja biasu, tryby, wariancja He, gradient numeryczny)
- `dane/` – oryginalny zbiór UCI; `figures/`, `results.json` – wyniki

Sieć 17-17-17-17-17-17-1 (5 warstw ukrytych, celowo głęboka, żeby problemy optymalizacji było widać).
Eksperymenty: punkt siodłowy na płaszczyźnie (GD vs ADAM), pełny zbiór vs paczki po 32 vs pojedyncze przykłady,
inicjalizacja N(0; 0,01) / N(0; 1) / He / N(0; 10), prosty gradient vs ADAM (start He i N(0; 0,01)).

Uruchomienie (z tego folderu, Windows):
```
py -m pip install numpy pandas matplotlib scikit-learn
py test_optim.py      # kilka sekund
py optim.py           # ok. 1,5 min, wykresy -> figures/, liczby -> results.json
py -m pip install notebook && py -m notebook raport_cw3_cz2.ipynb   # notatnik (Run All ok. 1,5 min)
```
Kod przygotowany z pomocą AI (Claude), oznaczony zgodnie z pkt 11 regulaminu.

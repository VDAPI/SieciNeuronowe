# Ćwiczenie 2 – regresja logistyczna (własna implementacja, NumPy)

- `raport_cw2.docx` / `raport_cw2.pdf` – raport do oddania
- `raport_cw2.ipynb` – ten sam raport jako notatnik Jupyter (opis + kod + wykresy, już uruchomiony)
- `PLAN_OMOWIENIA_cw2.md` – co mówić przy każdej stronie raportu + pytania prowadzącego
- `NAUKA_cw2.md/.docx` – teoria z wyprowadzeniami, objaśnienie kodu, pytania na wejściówkę
- `logreg.py` – model, uczenie i wszystkie eksperymenty; `data.py` – preprocessing (taki sam jak w ćw. 1)
- `test_logreg.py` – testy poprawności (gradient, sigmoida, metryki, zgodność minimum ze sklearn)
- `dane/` – oryginalny zbiór UCI; `figures/`, `results.json` – wyniki

Uruchomienie (z tego folderu, Windows):
```
py -m pip install numpy pandas matplotlib scikit-learn
py test_logreg.py     # ok. 10 s
py logreg.py          # ok. 3 min, wykresy -> figures/, liczby -> results.json
py -m pip install notebook && py -m notebook raport_cw2.ipynb   # notatnik (Run All ok. 1 min)
```
Kod przygotowany z pomocą AI (Claude), oznaczony zgodnie z pkt 11 regulaminu.

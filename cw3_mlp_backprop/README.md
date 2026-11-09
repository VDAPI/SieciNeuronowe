# Ćwiczenie 3 – sieć wielowarstwowa uczona propagacją wsteczną (własna implementacja, NumPy)

- `raport_cw3.ipynb` – raport jako notatnik Jupyter (opis + kod + wykresy, już uruchomiony)
- `raport_cw3.docx` – ten sam raport bez kodu (do oddania)
- `PLAN_OMOWIENIA_cw3.md` – co mówić przy każdym rozdziale raportu + pytania prowadzącego
- `NAUKA_cw3.md/.docx` – teoria z wyprowadzeniami, objaśnienie kodu, pytania na wejściówkę
- `mlp.py` – moduły Linear / ReLU / sigmoida + BCE (forward + backward), sieć MLP, uczenie i wszystkie eksperymenty
- `data.py` – preprocessing (taki sam jak w ćw. 1–2); `load_data()` w `mlp.py` dodaje wariant min-max i dane surowe
- `test_mlp.py` – testy poprawności (gradient każdego modułu i całej sieci vs numeryczny, 0 warstw = regresja z ćw. 2)
- `dane/` – oryginalny zbiór UCI; `figures/`, `results.json` – wyniki

Eksperymenty: szerokość warstwy ukrytej (2 / 4 / 17 / 68 / 256), dane surowe / min-max / standaryzowane,
liczba warstw ukrytych (0–4), dodatkowo inicjalizacja He. Wagi N(0; 0,01), α = 0,01, pełny batch.

Uruchomienie (z tego folderu, Windows):
```
py -m pip install numpy pandas matplotlib scikit-learn
py test_mlp.py        # kilka sekund
py mlp.py             # ok. 6 min, wykresy -> figures/, liczby -> results.json
py -m pip install notebook && py -m notebook raport_cw3.ipynb   # notatnik (Run All ok. 6 min)
```
Kod przygotowany z pomocą AI (Claude), oznaczony zgodnie z pkt 11 regulaminu.

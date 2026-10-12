# Sieci Neuronowe – rozwiązania ćwiczeń

Rozwiązania ćwiczeń laboratoryjnych z przedmiotu **Sieci Neuronowe**. Wszystkie ćwiczenia
korzystają ze zbioru **Heart Disease (Cleveland, UCI)** – binarna klasyfikacja obecności
choroby serca (303 pacjentów, 13 cech).

Każdy folder jest samodzielny: zawiera własną kopię danych (`dane/`), moduł
przygotowania danych (`data.py`), skrypt ćwiczenia, wygenerowane wykresy (`figures/`),
wyniki liczbowe (`results.json`), raport (`raport_cwX.md/.docx`) oraz materiał do nauki
(`NAUKA_cwX.md/.docx`).

## Ćwiczenia

| Nr | Temat | Folder |
|----|-------|--------|
| 1 | Analiza eksploracyjna zbioru i baseline (sklearn) | [`cw1_analiza_eksploracyjna/`](cw1_analiza_eksploracyjna/) |
| 2 | Regresja logistyczna – własna implementacja w NumPy | [`cw2_regresja_logistyczna/`](cw2_regresja_logistyczna/) |

## Uruchomienie

Wymagany Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

cd cw1_analiza_eksploracyjna
python eda.py          # wykresy -> figures/, liczby -> results.json

cd ../cw2_regresja_logistyczna
python logreg.py       # ok. 2,5 minuty; wykresy -> figures/, liczby -> results.json
```

Skrypty uruchamia się z folderu danego ćwiczenia; nadpisują one zawartość `figures/`
i `results.json`.

## Wykorzystanie AI

Kod w tym repozytorium powstał z pomocą AI (Claude), co jest oznaczone w plikach
źródłowych zgodnie z pkt 11 regulaminu przedmiotu.

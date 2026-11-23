# Ćwiczenie 4 – sieć z ćw. 3 w PyTorch + baseline na CIFAR10

- `raport_cw4.md` / `.docx` – raport: proces dochodzenia do zgodności z ćw. 3 (kroki K0–K9), porównanie wszystkich
  eksperymentów, wyjaśnienie pozostałych różnic, CIFAR10
- `torch_mlp.py` – model (`TorchMLP`, `from_numpy()` kopiuje wagi z ćw. 3), uczenie pełnym zbiorem (`fit_full`)
  i w trybach full / minibatch / single (`train_modes`), sampler odtwarzający kolejność z NumPy, metryki
- `equivalence.py` – kroki K0–K8: naiwny port → te same wagi → float64 → gradienty → koszt → krok optymalizatora
  → całe uczenie → kolejność paczek
- `replicate.py` – wszystkie eksperymenty z ćw. 3 (cz. 1 i 2) w PyTorch → `results_torch/*.json`, `figures/rep_*.png`
- `compare.py` – porównanie każdej liczby z `results.json` z ćw. 3 (kopie w `dane/cw3_*results.json`)
- `diagnose.py` – wyjaśnienie różnic: ReLU i `−∞·0 = NaN`, chaos przy danych surowych, ADAM (test z 1 ULP)
- `cifar10.py` – MLP 3072‑1024‑512‑256‑10 na CIFAR10 (baseline do ćw. 5) → `results_cifar10.json`,
  `figures/cifar_*.png`, `cifar10_mlp.pt`
- `raport_cw4_prezentacja.pdf` – krótki raport do prezentacji (styl jak `raport_cw3_krotki.pdf`);
  źródło `pdf/raport_cw4_prezentacja.html`, wykresy `figures/raport_*.png` z `wykresy_raport.py`
- `NAUKA_cw4.pdf` – materiał do nauki (wzór, intuicja, przykład liczbowy); źródło `pdf/NAUKA_cw4.html`
- `pdf/zbuduj_pdf.py` – HTML → PDF przez Edge headless (potrzebny internet: czcionka Carlito, KaTeX)
- `cw3_mlp.py`, `cw3_optim.py`, `data.py`, `dane/` – kod i dane z ćw. 3 bez zmian (punkt odniesienia)

Uruchomienie (z tego folderu; PyTorch CPU: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`):
```
python equivalence.py   # ok. 4 min
python replicate.py     # ok. 1 h; sekcje osobno/równolegle: python replicate.py baseline width ...
python diagnose.py      # ok. 3 min
python compare.py
python cifar10.py       # pobiera CIFAR10 (170 MB) do dane/cifar10
python wykresy_raport.py   # ok. 1 min, wykresy do raportu
python pdf/zbuduj_pdf.py   # PDF-y
```
Kod przygotowany z pomocą AI (Claude), oznaczony zgodnie z pkt 11 regulaminu.

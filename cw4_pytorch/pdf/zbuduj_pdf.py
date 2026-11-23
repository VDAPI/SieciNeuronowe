# -*- coding: utf-8 -*-
"""Składa PDF-y z plików HTML w tym folderze (Microsoft Edge w trybie headless, druk do PDF).

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu.]

Uruchomienie:  python zbuduj_pdf.py            -> ../raport_cw4_prezentacja.pdf, ../NAUKA_cw4.pdf
Wymaga internetu (czcionka Carlito i KaTeX z CDN).
"""
import os
import pathlib
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
DOCS = ["raport_cw4_prezentacja", "NAUKA_cw4"]


def build(name):
    src = HERE / f"{name}.html"
    dst = OUT / f"{name}.pdf"
    dst.unlink(missing_ok=True)
    profile = tempfile.mkdtemp(prefix="edge_pdf_")      # osobny profil: działa też przy otwartym Edge
    r = subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--no-first-run",
                    f"--user-data-dir={profile}",
                    "--virtual-time-budget=15000", "--run-all-compositor-stages-before-draw",
                    f"--print-to-pdf={dst}", src.as_uri()], capture_output=True, text=True, timeout=180)
    for _ in range(60):                 # Edge bywa, że kończy proces przed zapisaniem pliku
        if dst.exists() and dst.stat().st_size > 0:
            break
        time.sleep(0.5)
    if not dst.exists():
        sys.exit(f"Edge nie utworzył {dst}:\n{r.stderr[-2000:]}")
    print("zapisano", dst, f"({dst.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    for n in sys.argv[1:] or DOCS:
        if (HERE / f"{n}.html").exists():
            build(n)

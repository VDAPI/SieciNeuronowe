# -*- coding: utf-8 -*-
"""
SIECI NEURONOWE - ćwiczenie 4, cz. 3: większa sieć w pełni połączona (MLP) na CIFAR10 - baseline
do kolejnego ćwiczenia.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu należy go
 oznaczyć, zrozumieć i umieć samodzielnie wyjaśnić.]

CIFAR10: 60 000 kolorowych obrazków 32x32 (50 000 train, 10 000 test), 10 klas po równo.
Podział: 45 000 train / 5 000 walidacja (warstwowo, ziarno 42) / 10 000 test (oficjalny).
Wejście: obraz spłaszczony do 3072 = 3 * 32 * 32 liczb, standaryzacja per kanał (średnia i std z train).

Sieć: 3072 -> 1024 -> 512 -> 256 -> 10, ReLU (opcjonalnie dropout), wyjście = logity 10 klas,
koszt nn.CrossEntropyLoss (softmax + entropia krzyżowa w jednej funkcji - wieloklasowy odpowiednik
BCEWithLogitsLoss), ADAM lr = 1e-3, paczki po 128, wagi z epoki o najlepszej dokładności walidacyjnej.

Uruchomienie:  python cifar10.py      (pierwsze uruchomienie pobiera zbiór, ok. 170 MB, do dane/cifar10)
               wyniki -> results_cifar10.json, figures/cifar_*.png, cifar10_mlp.pt (wagi najlepszej sieci)
"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "dane", "cifar10")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
SEED = 42
EPOCHS = 40
BATCH = 128
LR = 1e-3
HIDDEN = (1024, 512, 256)
torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def load():
    tr = datasets.CIFAR10(ROOT, train=True, download=True)
    te = datasets.CIFAR10(ROOT, train=False, download=True)
    # .data to już tablica uint8 (N, 32, 32, 3) - szybciej niż ToTensor() obrazek po obrazku
    Xtr = torch.tensor(tr.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    Xte = torch.tensor(te.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    ytr, yte = torch.tensor(tr.targets), torch.tensor(te.targets)
    # warstwowy podział train / walidacja (500 obrazków z każdej klasy do walidacji)
    rng = np.random.default_rng(SEED)
    va_idx = np.concatenate([rng.permutation(np.where(ytr.numpy() == c)[0])[:500] for c in range(10)])
    mask = np.ones(len(ytr), bool); mask[va_idx] = False
    Xva, yva, Xtr, ytr = Xtr[va_idx], ytr[va_idx], Xtr[mask], ytr[mask]
    mu = Xtr.mean((0, 2, 3), keepdim=True); sd = Xtr.std((0, 2, 3), keepdim=True)     # statystyki tylko z train
    norm = lambda X: (X - mu) / sd                                                    # noqa: E731
    return (norm(Xtr), ytr), (norm(Xva), yva), (norm(Xte), yte), tr.classes, dict(mean=mu.flatten().tolist(),
                                                                                   std=sd.flatten().tolist())


class MLP(nn.Module):
    def __init__(self, hidden=HIDDEN, dropout=0.0, n_in=3 * 32 * 32, n_out=10):
        super().__init__()
        layers, prev = [nn.Flatten()], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else [])
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate(model, X, y, loss_fn, bs=2000):
    model.eval()                      # wyłącza dropout
    tot, correct, preds = 0.0, 0, []
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb, yb = X[i:i + bs].to(device), y[i:i + bs].to(device)
            out = model(xb)
            tot += float(loss_fn(out, yb)) * len(yb)
            p = out.argmax(1)
            correct += int((p == yb).sum()); preds.append(p.cpu())
    return tot / len(y), correct / len(y), torch.cat(preds)


def train_one(name, dropout, D, classes):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = D
    torch.manual_seed(SEED)
    model = MLP(dropout=dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    h = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "time"]}
    best = (-1.0, None, 0)
    t0 = time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss, run_ok, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
            run_loss += float(loss) * len(yb); run_ok += int((out.argmax(1) == yb).sum()); n += len(yb)
        vl, va, _ = evaluate(model, Xva, yva, loss_fn)
        h["train_loss"].append(run_loss / n); h["train_acc"].append(run_ok / n)
        h["val_loss"].append(vl); h["val_acc"].append(va); h["time"].append(time.perf_counter() - t0)
        if va > best[0]:
            best = (va, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
        print(f"[{name}] epoka {ep:2d}  train loss {run_loss / n:.3f} acc {run_ok / n:.3f} | "
              f"val loss {vl:.3f} acc {va:.3f} | {h['time'][-1]:.0f} s", flush=True)
    model.load_state_dict(best[1])
    tl, ta, pred = evaluate(model, Xte, yte, loss_fn)
    cm = np.zeros((10, 10), int)
    np.add.at(cm, (yte.numpy(), pred.numpy()), 1)
    res = dict(dropout=dropout, best_epoch=best[2], best_val_acc=best[0], test_loss=tl, test_acc=ta,
               per_class_acc={c: float(cm[i, i] / cm[i].sum()) for i, c in enumerate(classes)},
               confusion=cm.tolist(), history=h, n_params=sum(p.numel() for p in model.parameters()),
               time_s=h["time"][-1])
    print(f"[{name}] najlepsza epoka {best[2]}, val acc {best[0]:.4f}, TEST acc {ta:.4f}", flush=True)
    return model, res


def main():
    print(f"Using {device} device")
    t0 = time.perf_counter()
    *D, classes, stats = load()
    print(f"train {tuple(D[0][0].shape)}, val {tuple(D[1][0].shape)}, test {tuple(D[2][0].shape)}; "
          f"wczytanie {time.perf_counter() - t0:.1f} s")
    R = dict(settings=dict(hidden=list(HIDDEN), epochs=EPOCHS, batch=BATCH, lr=LR, optimizer="Adam", seed=SEED,
                           split="45000/5000/10000", normalization=stats, device=device), runs={})
    models = {}
    for name, p in [("MLP bez dropoutu", 0.0), ("MLP + dropout 0,3", 0.3)]:
        models[name], R["runs"][name] = train_one(name, p, D, classes)
    best_name = max(R["runs"], key=lambda k: R["runs"][k]["best_val_acc"])     # wybór po walidacji, nie po teście
    R["baseline"] = dict(model=best_name, test_acc=R["runs"][best_name]["test_acc"])
    torch.save(dict(state_dict=models[best_name].state_dict(), hidden=HIDDEN,
                    dropout=R["runs"][best_name]["dropout"], normalization=stats), os.path.join(HERE, "cifar10_mlp.pt"))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, r in R["runs"].items():
        e = range(1, len(r["history"]["train_loss"]) + 1)
        l1, = ax[0].plot(e, r["history"]["train_loss"], label=f"{name}: train")
        ax[0].plot(e, r["history"]["val_loss"], "--", c=l1.get_color(), label=f"{name}: walidacja")
        ax[1].plot(e, r["history"]["train_acc"], c=l1.get_color(), label=f"{name}: train")
        ax[1].plot(e, r["history"]["val_acc"], "--", c=l1.get_color(), label=f"{name}: walidacja")
    ax[0].set_xlabel("epoka"); ax[0].set_ylabel("entropia krzyżowa"); ax[0].set_title("CIFAR10, MLP 3072-1024-512-256-10")
    ax[1].set_xlabel("epoka"); ax[1].set_ylabel("dokładność"); ax[1].set_title("Dokładność")
    for a in ax:
        a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "cifar_1_curves.png"), dpi=130); plt.close()

    cm = np.array(R["runs"][best_name]["confusion"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="white" if cm[i, j] > 500 else "black")
    ax.set_xticks(range(10), classes, rotation=45, ha="right"); ax.set_yticks(range(10), classes)
    ax.set_xlabel("predykcja"); ax.set_ylabel("prawdziwa klasa")
    ax.set_title(f"{best_name}: test acc {R['baseline']['test_acc']:.3f}")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "cifar_2_confusion.png"), dpi=130); plt.close()

    with open(os.path.join(HERE, "results_cifar10.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

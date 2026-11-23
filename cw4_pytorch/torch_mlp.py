# -*- coding: utf-8 -*-
"""
SIECI NEURONOWE - ćwiczenie 4: sieć z ćwiczenia 3 odtworzona w PyTorch.

[KOD WYGENEROWANY Z POMOCĄ AI (Claude) - zgodnie z pkt 11 regulaminu należy go
 oznaczyć, zrozumieć i umieć samodzielnie wyjaśnić.]

Odpowiedniki modułów z listy 3:
  Linear (W: (n_out, n_in), y = X W^T + b)  -> nn.Linear (weight ma ten sam kształt (out, in), więc wagi
                                               kopiujemy bez transpozycji)
  ReLU                                      -> nn.ReLU
  SigmoidBCE (dL/dz = (p - y) / N)          -> nn.BCEWithLogitsLoss (sigmoida + BCE w jednej, stabilnej funkcji)
  backward() każdego modułu                 -> autograd: loss.backward() przechodzi graf obliczeniowy
  w <- w - alpha * dw   /  własny ADAM      -> torch.optim.SGD  /  torch.optim.Adam
  pętla po paczkach z np.permutation        -> DataLoader + sampler (ten sam albo torchowy generator losowy)

Wszystkie elementy, które mogą dać inny wynik niż implementacja NumPy, są parametrami
(dtype, sposób inicjalizacji, raportowana funkcja kosztu, kolejność przykładów) - patrz equivalence.py.
"""

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset

torch.set_num_threads(1)      # malutkie macierze (17 x 17): wątki tylko spowalniają

EPS_CLIP = 1e-12              # tak samo jak bce() w ćw. 3


# ----------------------------------------------------------------------------
# 1. Model
# ----------------------------------------------------------------------------
class MaskReLU(nn.Module):
    """ReLU zapisane jak w ćw. 3: x * [x > 0]. Różni się od nn.ReLU tylko dla x = -inf:
    -inf * 0 = NaN (IEEE 754), a max(0, -inf) = 0. Potrzebne tylko do odtworzenia rozbiegających się przebiegów."""

    def forward(self, x):
        return x * (x > 0)


class TorchMLP(nn.Module):
    """Linear -> ReLU -> ... -> Linear(., 1). forward zwraca logit z (kształt (N,)).

    init: 'normal'  - W ~ N(0; init_scale), b = 0           (jak ćw. 3)
          'he'      - W ~ N(0; sqrt(2 / n_in)), b = 0        (jak ćw. 3)
          'torch'   - domyślna inicjalizacja nn.Linear (Kaiming-uniform, bias też losowy)
    Wagi losowane torchowym generatorem - inne liczby niż w NumPy przy tym samym ziarnie.
    Żeby mieć IDENTYCZNE wagi, używamy from_numpy()."""

    def __init__(self, n_in, hidden=(17,), init="normal", init_scale=0.01, seed=0, dtype=torch.float64, relu="torch"):
        super().__init__()
        torch.manual_seed(seed)           # dla init='torch' (nn.Linear losuje przy tworzeniu)
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU() if relu == "torch" else MaskReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers).to(dtype)
        self.linears = [m for m in self.net if isinstance(m, nn.Linear)]
        if init != "torch":
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                for lin in self.linears:
                    std = (2.0 / lin.in_features) ** 0.5 if init == "he" else init_scale
                    lin.weight.normal_(0.0, std, generator=g)
                    lin.bias.zero_()

    def forward(self, x):
        return self.net(x).squeeze(-1)

    @classmethod
    def from_numpy(cls, np_model, dtype=torch.float64, relu="torch"):
        """Ta sama architektura i dokładnie te same wagi co obiekt MLP z ćw. 3."""
        lins = np_model.linears
        hidden = tuple(l.W.shape[0] for l in lins[:-1])
        m = cls(lins[0].W.shape[1], hidden, dtype=dtype, relu=relu)
        m.load_numpy(np_model)
        return m

    def load_numpy(self, np_model):
        with torch.no_grad():
            for lt, ln in zip(self.linears, np_model.linears):
                lt.weight.copy_(torch.from_numpy(ln.W))     # (n_out, n_in) - ten sam układ, bez .T
                lt.bias.copy_(torch.from_numpy(ln.b))
        return self

    @property
    def dtype(self):
        return self.linears[0].weight.dtype

    def get_params(self):
        return {k: v.detach().clone() for k, v in self.state_dict().items()}

    def set_params(self, params):
        self.load_state_dict(params)

    def predict_proba(self, X):
        with torch.no_grad():
            return torch.sigmoid(self(X))


# ----------------------------------------------------------------------------
# 2. Koszt i metryki
# ----------------------------------------------------------------------------
loss_fn = nn.BCEWithLogitsLoss()      # do uczenia: gradient dokładnie (sigma(z) - y) / N, jak SigmoidBCE.backward


def bce_report(z, y, mode="clip"):
    """Wartość kosztu do raportowania (bez gradientu).
    'clip'   - jak bce() w ćw. 3: p obcięte do [1e-12, 1 - 1e-12] (maks. 27,6 na przykład)
    'native' - BCEWithLogitsLoss liczony z logitu, bez obcinania (dokładny)."""
    with torch.no_grad():
        if mode == "native":
            return float(loss_fn(z, y))
        z, y = z.double(), y.double()        # 1 - 1e-12 w float32 to po prostu 1.0
        p = torch.sigmoid(z).clamp(EPS_CLIP, 1 - EPS_CLIP)
        return float(-(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean())


def metrics(y, p, thr=0.5):
    yhat = p >= thr
    pos = y == 1
    tp = int((yhat & pos).sum()); fp = int((yhat & ~pos).sum())
    fn = int((~yhat & pos).sum()); tn = int((~yhat & ~pos).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=prec, recall=rec, f1=f1, accuracy=(tp + tn) / len(y), tp=tp, fp=fp, fn=fn, tn=tn)


def to_t(*arrays, dtype=torch.float64):
    return [torch.as_tensor(a, dtype=dtype) for a in arrays]


def grad_norm(lin, with_bias=True):
    s = lin.weight.grad.pow(2).sum()
    if with_bias:
        s = s + lin.bias.grad.pow(2).sum()
    return float(s.sqrt())


# ----------------------------------------------------------------------------
# 3. Uczenie pełnym zbiorem - odpowiednik MLP.fit z ćw. 3 cz. 1
# ----------------------------------------------------------------------------
def fit_full(model, X, y, Xv, yv, lr=0.01, epochs=1000, log_every=10, report="clip"):
    """Pełny batch, SGD. Ta sama kolejność co w ćw. 3: koszt i gradient dla wag sprzed kroku,
    koszt walidacyjny dla tych samych wag, najlepsze wagi wg kosztu walidacyjnego, potem krok."""
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    h = {k: [] for k in ["loss", "val_loss", "log_epoch", "f1", "val_f1", "val_precision", "val_recall", "grad_norms"]}
    best = (np.inf, None, 0)
    for ep in range(epochs):
        opt.zero_grad()                       # backward AKUMULUJE (+=) gradient w .grad - trzeba zerować
        z = model(X)
        loss_fn(z, y).backward()              # autograd: przejście wstecz po grafie obliczeniowym
        zd = z.detach()
        loss = bce_report(zd, y, report)
        h["loss"].append(loss)
        with torch.no_grad():
            zv = model(Xv)
        h["val_loss"].append(bce_report(zv, yv, report))
        if h["val_loss"][-1] < best[0] - 1e-9:
            best = (h["val_loss"][-1], model.get_params(), ep)
        if ep % log_every == 0 or ep == epochs - 1:
            h["log_epoch"].append(ep)
            h["f1"].append(metrics(y, torch.sigmoid(zd))["f1"])
            h["grad_norms"].append([grad_norm(l) for l in model.linears])
            mv = metrics(yv, torch.sigmoid(zv))
            h["val_f1"].append(mv["f1"]); h["val_precision"].append(mv["precision"]); h["val_recall"].append(mv["recall"])
        if not np.isfinite(loss):
            break
        opt.step()
    model.history = h
    model.best_val_loss, model.best_params, model.best_epoch = best
    return model


def epochs_to(loss_hist, target):
    idx = np.where(np.asarray(loss_hist) <= target)[0]
    return int(idx[0]) if len(idx) else None


def summarize_full(m, data, plateau=0.6):
    """Te same liczby co summarize() w ćw. 3 cz. 1."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    L = m.history["loss"]
    final = dict(train=metrics(ytr, m.predict_proba(Xtr)), test=metrics(yte, m.predict_proba(Xte)))
    last = m.get_params()
    m.set_params(m.best_params)
    early = dict(val=metrics(yva, m.predict_proba(Xva)), test=metrics(yte, m.predict_proba(Xte)))
    m.set_params(last)
    return dict(final_train_loss=float(L[-1]), min_train_loss=float(np.nanmin(L)),
                final_val_loss=float(m.history["val_loss"][-1]), best_val_loss=float(m.best_val_loss),
                best_epoch=int(m.best_epoch), plateau_exit=epochs_to(L, plateau),
                diverged=bool(not np.isfinite(L[-1])), final=final, early_stopping=early)


# ----------------------------------------------------------------------------
# 4. Uczenie w trybach full / minibatch / single - odpowiednik train() z ćw. 3 cz. 2
# ----------------------------------------------------------------------------
class NumpyPermutationBatches(Sampler):
    """Paczki indeksów w DOKŁADNIE tej kolejności co w ćw. 3 cz. 2:
    co epokę rng.permutation(n) z np.random.default_rng(seed) (tylko gdy paczka < n)."""

    def __init__(self, n, batch_size, seed):
        self.n, self.bs, self.rng = n, batch_size, np.random.default_rng(seed)

    def __iter__(self):
        order = self.rng.permutation(self.n) if self.bs < self.n else np.arange(self.n)
        for s in range(0, self.n, self.bs):
            yield torch.as_tensor(order[s:s + self.bs])

    def __len__(self):
        return -(-self.n // self.bs)


def make_loader(X, y, batch_size, seed, order="numpy"):
    ds = TensorDataset(X, y)
    if order == "numpy":
        # batch_size=None: sampler zwraca od razu całą paczkę indeksów, TensorDataset[indeksy] -> paczka
        return DataLoader(ds, sampler=NumpyPermutationBatches(len(y), batch_size, seed), batch_size=None)
    g = torch.Generator().manual_seed(seed)  # natywnie: losowa kolejność z generatora torcha
    return DataLoader(ds, batch_size=batch_size, shuffle=batch_size < len(y), generator=g)


def batch_size_for(mode, n, batch=32):
    return {"full": n, "minibatch": batch, "single": 1}[mode]


def train_modes(model, opt, X, y, Xv, yv, mode="full", batch=32, epochs=500, seed=42, log_every=1,
                order="numpy", report="clip"):
    """Jedna epoka = jedno przejście przez DataLoader. Koszt train/val na całych zbiorach po epoce."""
    n = len(y)
    loader = make_loader(X, y, batch_size_for(mode, n, batch), seed, order)
    h = {k: [] for k in ["epoch", "step", "loss", "val_loss", "val_f1", "grad_norms", "time"]}
    best = (np.inf, None, 0)
    steps, t0 = 0, time.perf_counter()

    def log(ep):
        opt.zero_grad()
        z = model(X)
        loss_fn(z, y).backward()                    # pełny gradient tylko do pomiaru normy
        loss = bce_report(z.detach(), y, report)
        with torch.no_grad():
            zv = model(Xv)
        h["epoch"].append(ep); h["step"].append(steps); h["loss"].append(loss)
        h["val_loss"].append(bce_report(zv, yv, report)); h["val_f1"].append(metrics(yv, torch.sigmoid(zv))["f1"])
        h["grad_norms"].append([grad_norm(l, with_bias=False) for l in model.linears])
        h["time"].append(time.perf_counter() - t0)
        opt.zero_grad()
        return loss

    log(0)
    for ep in range(1, epochs + 1):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
            steps += 1
        if ep % log_every == 0 or ep == epochs:
            loss = log(ep)
            if h["val_loss"][-1] < best[0] - 1e-9:
                best = (h["val_loss"][-1], model.get_params(), ep)
            if not np.isfinite(loss):
                break
    model.history = h
    model.best_val_loss, model.best_params, model.best_epoch = best
    return model


def epochs_to_h(hist, key, target):
    for ep, value in zip(hist["epoch"], hist[key]):
        if value <= target:
            return int(ep)
    return None


def summarize_modes(m, data, plateau=0.6, target=0.3):
    """Te same liczby co summarize() w ćw. 3 cz. 2."""
    (Xtr, ytr), (Xva, yva), (Xte, yte) = data
    h = m.history
    out = dict(start_loss=float(h["loss"][0]), final_loss=float(h["loss"][-1]),
               min_loss=float(np.nanmin(h["loss"])), diverged=bool(not np.isfinite(h["loss"][-1])),
               epochs_run=int(h["epoch"][-1]), steps=int(h["step"][-1]), time_s=float(h["time"][-1]),
               plateau_exit=epochs_to_h(h, "loss", plateau), epochs_to_target=epochs_to_h(h, "loss", target),
               best_val_loss=float(m.best_val_loss), best_epoch=int(m.best_epoch), test_f1_early=None)
    if m.best_params is not None:
        last = m.get_params()
        m.set_params(m.best_params)
        out["test_early"] = metrics(yte, m.predict_proba(Xte))
        out["test_f1_early"] = out["test_early"]["f1"]
        m.set_params(last)
    if out["epochs_to_target"] is not None:
        k = h["epoch"].index(out["epochs_to_target"])
        out["steps_to_target"], out["time_to_target"] = int(h["step"][k]), float(h["time"][k])
    return out


def activation_stds(model, X):
    """std wyjścia każdej warstwy liniowej - zbierane hookami (forward hook), bez przepisywania forward()."""
    stds = []
    hooks = [l.register_forward_hook(lambda mod, inp, out: stds.append(float(out.std(unbiased=False))))
             for l in model.linears]
    with torch.no_grad():
        model(X)
    for hk in hooks:
        hk.remove()
    return stds

"""Training / evaluation for the MLP, GCN and semi-supervised GAN."""
from __future__ import annotations

import copy
import random
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from config import CLASS_NAMES, Config
from models import (Generator, ModalityAE, MLPClassifier, MultimodalGCN, SGANDiscriminator, concat)
from preprocessing import Bundle

K = len(CLASS_NAMES)


# helpers
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensors(b: Bundle, mods: List[str], dev) -> Dict[str, torch.Tensor]:
    return {m: torch.as_tensor(b.X[m], dtype=torch.float32, device=dev) for m in mods}


def class_weights(y_train: np.ndarray, dev) -> torch.Tensor:
    cnt = np.bincount(y_train, minlength=K).astype(float)
    w = np.where(cnt > 0, cnt.sum() / (K * np.maximum(cnt, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32, device=dev)


def metrics(y_true: np.ndarray, logits: torch.Tensor) -> Dict:
    prob = F.softmax(logits, dim=1).detach().cpu().numpy()
    pred = prob.argmax(1)
    m = {"acc": accuracy_score(y_true, pred),
         "balanced_acc": balanced_accuracy_score(y_true, pred),
         "macro_f1": f1_score(y_true, pred, average="macro", labels=list(range(K)), zero_division=0)}
    try:
        present = np.unique(y_true)
        m["auc_ovr"] = roc_auc_score(y_true, prob[:, present] / prob[:, present].sum(1, keepdims=True),
                                     multi_class="ovr", average="macro", labels=present) if len(present) > 2 else float("nan")
    except ValueError:
        m["auc_ovr"] = float("nan")
    m["confusion"] = confusion_matrix(y_true, pred, labels=list(range(K))).tolist()
    return m


class BestKeeper:
    def __init__(self, metric: str):
        self.metric, self.best, self.state, self.epoch = metric, -1.0, None, -1

    def update(self, model: nn.Module, val_metrics: Dict, epoch: int):
        if val_metrics[self.metric] > self.best:
            self.best, self.epoch = val_metrics[self.metric], epoch
            self.state = copy.deepcopy(model.state_dict())

    def restore(self, model: nn.Module):
        if self.state is not None:
            model.load_state_dict(self.state)


#  stage 1: dimensionality reduction
def pretrain_encoders(b: Bundle, mods: List[str], cfg: Config, dev) -> nn.ModuleDict:
    """Train one autoencoder per modality on TRAIN subjects only (labels unused). Returns the encoders."""
    encoders = nn.ModuleDict()
    tr = torch.as_tensor(b.idx["train"], device=dev)
    for m in mods:
        x = torch.as_tensor(b.X[m], device=dev)[tr]
        ae = ModalityAE(x.shape[1], cfg.ae_hidden, cfg.latent_dim).to(dev)
        opt = torch.optim.Adam(ae.parameters(), lr=cfg.ae_lr, weight_decay=cfg.weight_decay)
        bs = min(cfg.batch_size, len(x))
        for _ in range(cfg.ae_epochs):
            ae.train()
            perm = torch.randperm(len(x), device=dev)
            for i in range(0, len(x) - 1, bs):
                xb = x[perm[i:i + bs]]
                if len(xb) < 2:
                    continue
                rec, _ = ae(xb)
                loss = F.mse_loss(rec, xb)
                opt.zero_grad(); loss.backward(); opt.step()
        encoders[m] = ae.encoder
    return encoders


@torch.no_grad()
def embed_all(encoders: nn.ModuleDict, x: Dict[str, torch.Tensor]) -> torch.Tensor:
    encoders.eval()
    z = [F.normalize(encoders[m](x[m]), dim=1) for m in encoders]      # each modality unit-norm -> equal weight
    return F.normalize(torch.cat(z, dim=1), dim=1)


def build_graph(emb: torch.Tensor, k: int, weighted: bool) -> torch.Tensor:
    """Patient-similarity graph: pairwise cosine -> top-K neighbours per patient -> symmetrise -> D^-1/2 (A+I) D^-1/2.
    Uses features only (no labels)."""
    n = emb.shape[0]
    sim = emb @ emb.T
    sim.fill_diagonal_(-float("inf"))
    k = min(k, n - 1)
    top = sim.topk(k, dim=1)
    a = torch.zeros(n, n, device=emb.device)
    w = top.values.clamp(min=0) if weighted else torch.ones_like(top.values)
    a.scatter_(1, top.indices, w)
    a = torch.maximum(a, a.T)
    a = a + torch.eye(n, device=emb.device)
    d = a.sum(1).pow(-0.5)
    return d[:, None] * a * d[None, :]


# MLP
def train_mlp(b: Bundle, mods: List[str], cfg: Config, dev, seed: int) -> Dict:
    set_seed(seed)
    x = to_tensors(b, mods, dev)
    y = torch.as_tensor(b.y, device=dev)
    tr, va, te = [torch.as_tensor(b.idx[s], device=dev) for s in ("train", "val", "test")]
    model = MLPClassifier({m: x[m].shape[1] for m in mods}, cfg.mlp_hidden, K, cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.mlp_lr, weight_decay=cfg.weight_decay)
    w = class_weights(b.y[b.idx["train"]], dev) if cfg.use_class_weights else None
    keeper, hist = BestKeeper(cfg.select_metric), []
    bs = cfg.batch_size
    for ep in range(cfg.mlp_epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            loss = F.cross_entropy(model({m: x[m][idx] for m in mods}), y[idx], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vm = metrics(b.y[b.idx["val"]], model({m: x[m][va] for m in mods}))
        keeper.update(model, vm, ep); hist.append({"epoch": ep, "loss": float(loss), **{k: vm[k] for k in ("acc", "macro_f1")}})
    keeper.restore(model); model.eval()
    with torch.no_grad():
        full = model(x)
    return _result("mlp", model, x, b, full, keeper, hist,
                   explain_forward=lambda xd, node: model({m: xd[m][node:node + 1] for m in mods})[0])


# GCN
def train_gcn(b: Bundle, mods: List[str], cfg: Config, dev, seed: int) -> Dict:
    set_seed(seed)
    x = to_tensors(b, mods, dev)
    y = torch.as_tensor(b.y, device=dev)
    tr, va, te = [torch.as_tensor(b.idx[s], device=dev) for s in ("train", "val", "test")]

    encoders = pretrain_encoders(b, mods, cfg, dev)                      # stage 1: -> latent_dim per modality
    a_hat = build_graph(embed_all(encoders, x), cfg.graph_k, cfg.graph_weighted)   # stage 2: patient graph
    model = MultimodalGCN({m: x[m].shape[1] for m in mods}, K, cfg.latent_dim, cfg.ae_hidden, cfg.gcn_hidden,
                          cfg.dropout, pretrained_encoders=copy.deepcopy(encoders)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.gcn_lr, weight_decay=cfg.weight_decay)
    w = class_weights(b.y[b.idx["train"]], dev) if cfg.use_class_weights else None
    keeper, hist = BestKeeper(cfg.select_metric), []
    for ep in range(cfg.gcn_epochs):
        model.train()
        loss = F.cross_entropy(model(x, a_hat, tr), y[tr], weight=w)     # only TRAIN labels are used
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vm = metrics(b.y[b.idx["val"]], model(x, a_hat, va))
        keeper.update(model, vm, ep); hist.append({"epoch": ep, "loss": float(loss), **{k: vm[k] for k in ("acc", "macro_f1")}})
    keeper.restore(model); model.eval()
    with torch.no_grad():
        full = model(x, a_hat)
    res = _result("gcn", model, x, b, full, keeper, hist,
                  explain_forward=lambda xd, node: model(xd, a_hat, torch.tensor([node], device=dev))[0])
    res["graph_edges"] = int(((a_hat > 0).sum() - a_hat.shape[0]) // 2)
    return res


#  GAN
def train_gan(b: Bundle, mods: List[str], cfg: Config, dev, seed: int) -> Dict:
    """Semi-supervised GAN.  The generator's samples are ONLY used as 'fake' negatives for the discriminator's
    real/fake loss and are never added to any dataset or used for evaluation. Classification = D's K logits."""
    set_seed(seed)
    x = to_tensors(b, mods, dev)
    xf = concat(x, mods)
    y = torch.as_tensor(b.y, device=dev)
    tr, va = [torch.as_tensor(b.idx[s], device=dev) for s in ("train", "val")]
    unl = torch.as_tensor(np.concatenate([b.idx["train"], b.idx["val"], b.idx["test"]]) if cfg.gan_use_heldout_unlabeled
                          else b.idx["train"], device=dev)

    D = SGANDiscriminator({m: x[m].shape[1] for m in mods}, cfg.gan_hidden, K, cfg.dropout).to(dev)
    G = Generator(cfg.gan_z_dim, cfg.gan_hidden, xf.shape[1]).to(dev)
    oD = torch.optim.Adam(D.parameters(), lr=cfg.gan_lr, betas=(0.5, 0.999), weight_decay=cfg.weight_decay)
    oG = torch.optim.Adam(G.parameters(), lr=cfg.gan_lr, betas=(0.5, 0.999))
    w = class_weights(b.y[b.idx["train"]], dev) if cfg.use_class_weights else None
    keeper, hist = BestKeeper(cfg.select_metric), []
    bs = min(cfg.batch_size, len(tr))

    for ep in range(cfg.gan_epochs):
        D.train(); G.train()
        perm = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            xl, yl = xf[idx], y[idx]
            xu = xf[unl[torch.randint(len(unl), (len(idx),), device=dev)]]
            fake = G(torch.randn(len(idx), cfg.gan_z_dim, device=dev))

            # ---- discriminator: supervised CE + unsupervised real/fake
            l_sup = F.cross_entropy(D(xl), yl, weight=w)
            lse_r = torch.logsumexp(D(xu), 1)
            lse_f = torch.logsumexp(D(fake.detach()), 1)
            l_uns = -(lse_r - F.softplus(lse_r)).mean() + F.softplus(lse_f).mean()
            lossD = l_sup + cfg.gan_unsup_weight * l_uns
            oD.zero_grad(); lossD.backward(); oD.step()

            # ---- generator: feature matching
            with torch.no_grad():
                _, f_real = D(xu, return_features=True)
            _, f_fake = D(G(torch.randn(len(idx), cfg.gan_z_dim, device=dev)), return_features=True)
            lossG = F.mse_loss(f_fake.mean(0), f_real.mean(0))
            oG.zero_grad(); lossG.backward(); oG.step()

        D.eval()
        with torch.no_grad():
            vm = metrics(b.y[b.idx["val"]], D(xf[va]))
        keeper.update(D, vm, ep)
        hist.append({"epoch": ep, "loss_D": float(lossD), "loss_G": float(lossG), "acc": vm["acc"], "macro_f1": vm["macro_f1"]})
    keeper.restore(D); D.eval()
    with torch.no_grad():
        full = D(xf)

    sizes = [x[m].shape[1] for m in mods]
    def fwd(xd, node):
        return D(concat({m: xd[m][node:node + 1] for m in mods}, mods))[0]
    return _result("gan", D, x, b, full, keeper, hist, explain_forward=fwd)


#  common
def _result(name, model, x, b: Bundle, full_logits: torch.Tensor, keeper: BestKeeper, hist, explain_forward: Callable):
    out = {"name": name, "model": model, "x": x, "history": hist, "best_epoch": keeper.epoch,
           "explain_forward": explain_forward}
    for split in ("train", "val", "test"):
        i = b.idx[split]
        out[split] = metrics(b.y[i], full_logits[torch.as_tensor(i, device=full_logits.device)])
    return out


TRAINERS = {"mlp": train_mlp, "gcn": train_gcn, "gan": train_gan}

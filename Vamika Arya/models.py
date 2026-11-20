from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def concat(x: Dict[str, torch.Tensor], mods: Sequence[str]) -> torch.Tensor:
    return torch.cat([x[m] for m in mods], dim=1)


# MLP baseline
class MLPClassifier(nn.Module):
    def __init__(self, in_dims: Dict[str, int], hidden: List[int], n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.mods = list(in_dims)
        layers, d = [], sum(in_dims.values())
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(d, n_classes)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.out(self.body(concat(x, self.mods)))


#  encoder / autoencoder
class Encoder(nn.Module):
    """LayerNorm (not BatchNorm) so that nodes of the full graph never share batch statistics."""

    def __init__(self, d_in: int, hidden: int, latent: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, latent))

    def forward(self, x):
        return self.net(x)


class ModalityAE(nn.Module):
    def __init__(self, d_in: int, hidden: int, latent: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = Encoder(d_in, hidden, latent, dropout)
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, d_in))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


#GCN
class GCNLayer(nn.Module):
    """H' = A_hat @ H @ W   with A_hat = D^-1/2 (A + I) D^-1/2 precomputed (dense, N x N)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out, bias=False)
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, h, a_hat):
        return a_hat @ self.lin(h) + self.bias


class MultimodalGCN(nn.Module):
    def __init__(self, in_dims: Dict[str, int], n_classes: int, latent: int = 32, enc_hidden: int = 256,
                 gcn_hidden: int = 64, dropout: float = 0.3, pretrained_encoders: Optional[nn.ModuleDict] = None):
        super().__init__()
        self.mods = list(in_dims)
        self.encoders = pretrained_encoders or nn.ModuleDict(
            {m: Encoder(d, enc_hidden, latent, dropout) for m, d in in_dims.items()})
        d = latent * len(self.mods)
        self.gcn1 = GCNLayer(d, gcn_hidden)             # neighbour aggregation
        self.gcn2 = GCNLayer(gcn_hidden, gcn_hidden)    # higher-order representation
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(gcn_hidden, gcn_hidden), nn.BatchNorm1d(gcn_hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(gcn_hidden, n_classes))

    def embed(self, x):
        return torch.cat([self.encoders[m](x[m]) for m in self.mods], dim=1)

    def forward(self, x: Dict[str, torch.Tensor], a_hat: torch.Tensor, idx: Optional[torch.Tensor] = None):
       
        h = self.embed(x)
        h = self.drop(F.relu(self.gcn1(self.drop(h), a_hat)))
        h = F.relu(self.gcn2(h, a_hat))
        if idx is not None:
            h = h[idx]
        return self.head(h)


# ------------------------------------------------------------------------------------------------ semi-supervised GAN
class Generator(nn.Module):
    def __init__(self, z_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(),
                                 nn.Linear(hidden, out_dim))

    def forward(self, z):
        return self.net(z)


class SGANDiscriminator(nn.Module):
    

    def __init__(self, in_dims: Dict[str, int], hidden: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.mods = list(in_dims)
        d = sum(in_dims.values())
        self.features = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, hidden), nn.LeakyReLU(0.2), nn.Dropout(dropout),
                                      nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.2), nn.Dropout(dropout))
        self.cls = nn.Linear(hidden // 2, n_classes)

    def forward(self, x, return_features: bool = False):
        flat = x if torch.is_tensor(x) else concat(x, self.mods)
        f = self.features(flat)
        out = self.cls(f)
        return (out, f) if return_features else out

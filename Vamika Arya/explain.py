from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from config import CLASS_NAMES, Config
from preprocessing import Bundle


def integrated_gradients(forward, inputs: Dict[str, torch.Tensor], node: int, target: int, steps: int = 50):
    """forward(x_dict, node) -> logits (C,).  Returns {modality: attribution row (d,)} for `node`'s own features."""
    total = {m: torch.zeros_like(v) for m, v in inputs.items()}
    for s in range(steps):
        a = (s + 0.5) / steps                                           # midpoint rule
        xs = {m: (a * v).detach().requires_grad_(True) for m, v in inputs.items()}
        out = forward(xs, node)[target]
        grads = torch.autograd.grad(out, list(xs.values()), allow_unused=True)
        for m, g in zip(xs, grads):
            if g is not None:
                total[m] += g
    return {m: (inputs[m][node] * total[m][node] / steps).detach().cpu().numpy() for m in inputs}


def explain(result: Dict, b: Bundle, mods: List[str], cfg: Config, out_dir: Path, tag: str,
            target_class: str = "AD") -> pd.DataFrame:
    """IG on TEST subjects whose true class is `target_class`; attribution of the target-class logit."""
    model = result["model"]; model.eval()
    t = CLASS_NAMES.index(target_class)
    nodes = [int(i) for i in b.idx["test"] if b.y[i] == t][:cfg.ig_max_subjects]
    if not nodes:
        print(f"[xai] no test subjects of class {target_class}; skipped")
        return pd.DataFrame()
    acc = {m: [] for m in mods}
    for n in nodes:
        attr = integrated_gradients(result["explain_forward"], result["x"], n, t, cfg.ig_steps)
        for m in mods:
            acc[m].append(attr[m])
    rows = []
    for m in mods:
        a = np.stack(acc[m])
        for j, name in enumerate(b.names[m]):
            rows.append({"modality": m, "feature": name, "mean_abs_attr": np.abs(a[:, j]).mean(),
                         "mean_attr": a[:, j].mean()})
    df = pd.DataFrame(rows)
    df["modality_share"] = df.groupby("modality")["mean_abs_attr"].transform("sum") / df["mean_abs_attr"].sum()
    df = df.sort_values("mean_abs_attr", ascending=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"ig_{tag}.csv", index=False)
    share = df.groupby("modality")["mean_abs_attr"].sum() / df["mean_abs_attr"].sum()
    print(f"[xai] {tag}: {len(nodes)} {target_class} test subjects | modality share: "
          + ", ".join(f"{m}={v:.2f}" for m, v in share.items()))
    print(df.head(10)[["modality", "feature", "mean_abs_attr", "mean_attr"]].to_string(index=False))
    return df

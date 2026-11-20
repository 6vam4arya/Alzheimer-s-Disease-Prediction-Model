import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from config import CLASS_NAMES, Config
from explain import explain
from preprocessing import build_bundle, load_raw
from training import TRAINERS, get_device

SINGLE_NAMES = {"fmri": "imaging_only", "clinical": "clinical_only", "snp": "snp_only"}


def modality_sets(cfg: Config):
    """Single-modality runs + one run with all modalities in cfg.modalities."""
    sets = {SINGLE_NAMES[m]: [m] for m in cfg.modalities}
    if len(cfg.modalities) > 1:
        sets["multimodal"] = list(cfg.modalities)
    return sets


def main():
    cfg = Config()
    MODALITY_SETS = modality_sets(cfg)
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["mlp", "gcn", "gan"], choices=list(TRAINERS))
    ap.add_argument("--sets", nargs="+", default=list(MODALITY_SETS), choices=list(MODALITY_SETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--explain", action="store_true", help="integrated gradients for AD test subjects (multimodal)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.seeds is not None:
        cfg.seeds = tuple(args.seeds)
    out = Path(args.out or cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = get_device(); print(f"device: {dev}")

    raw = load_raw(cfg)
    records, curves = [], {}
    for seed in cfg.seeds:
        bundle = build_bundle(raw, cfg, seed)
        for set_name in args.sets:
            mods = MODALITY_SETS[set_name]
            for mname in args.models:
                res = TRAINERS[mname](bundle, mods, cfg, dev, seed)
                t = res["test"]
                print(f"seed={seed} {mname:>3} {set_name:<13} val_f1={res['val']['macro_f1']:.3f} "
                      f"test acc={t['acc']:.3f} bal_acc={t['balanced_acc']:.3f} f1={t['macro_f1']:.3f} "
                      f"auc={t['auc_ovr']:.3f} (best epoch {res['best_epoch']})")
                records.append({"seed": seed, "model": mname, "modalities": set_name,
                                **{f"test_{k}": v for k, v in t.items() if k != "confusion"},
                                "val_macro_f1": res["val"]["macro_f1"], "best_epoch": res["best_epoch"],
                                "confusion": json.dumps(t["confusion"])})
                curves[f"{mname}_{set_name}_seed{seed}"] = res["history"]
                if args.explain and set_name == "multimodal" and seed == cfg.seeds[0]:
                    explain(res, bundle, mods, cfg, out, f"{mname}_{set_name}_seed{seed}")

    df = pd.DataFrame(records)
    df.to_csv(out / "results_per_seed.csv", index=False)
    json.dump(curves, open(out / "training_curves.json", "w"))
    json.dump({k: str(v) for k, v in asdict(cfg).items()}, open(out / "config_used.json", "w"), indent=2)

    cols = ["test_acc", "test_balanced_acc", "test_macro_f1", "test_auc_ovr"]
    summ = df.groupby(["model", "modalities"])[cols].agg(["mean", "std"]).round(4)
    summ.to_csv(out / "summary.csv")
    print("\n=== test results (mean/std over seeds) ===")
    print(summ.to_string())

    if {"multimodal"} <= set(df.modalities):
        print("\n=== improvement of multimodal over best single modality (macro-F1, mean over seeds) ===")
        for m in df.model.unique():
            mean = df[df.model == m].groupby("modalities")["test_macro_f1"].mean()
            single = mean.drop("multimodal", errors="ignore")
            if len(single):
                print(f"{m}: multimodal={mean['multimodal']:.3f}  best single ({single.idxmax()})={single.max():.3f}  "
                      f"delta={mean['multimodal'] - single.max():+.3f}")
    print(f"\nsaved to {out.resolve()}")


if __name__ == "__main__":
    main()

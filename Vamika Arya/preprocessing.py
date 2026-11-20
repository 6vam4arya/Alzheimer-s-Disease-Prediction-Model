from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal, stats
from sklearn.compose import ColumnTransformer
from sklearn.covariance import LedoitWolf, graphical_lasso
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CLASS_NAMES, Config


# =====================================================================================================================
# fMRI
# =====================================================================================================================
def _read_timeseries(path: Path, roi_by_time: bool) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path).astype(float)
    else:
        df = pd.read_csv(path, sep=None, engine="python", header=None)
        df = df.apply(pd.to_numeric, errors="coerce")
        if df.iloc[0].isna().all():          # header row
            df = df.iloc[1:]
        arr = df.to_numpy(dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D array, got shape {arr.shape}")
    return arr.T if roi_by_time else arr


def clean_timeseries(ts: np.ndarray, cfg: Config) -> np.ndarray:
    """Linear detrend, optional band-pass, z-score each ROI ("standardise data" in your notes)."""
    ts = np.nan_to_num(ts, nan=0.0)
    ts = signal.detrend(ts, axis=0, type="linear")
    if cfg.fmri_bandpass and cfg.fmri_tr:
        lo, hi = cfg.fmri_bandpass
        b, a = signal.butter(2, [lo, hi], btype="band", fs=1.0 / cfg.fmri_tr)
        if ts.shape[0] > 3 * max(len(a), len(b)):
            ts = signal.filtfilt(b, a, ts, axis=0)
    sd = ts.std(axis=0, keepdims=True)
    return (ts - ts.mean(axis=0, keepdims=True)) / np.where(sd < 1e-8, 1.0, sd)


def _precision_to_partial(prec: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(prec), 1e-12, None))
    pc = -prec / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pc


def connectivity_vector(ts: np.ndarray, method: str, glasso_alpha: float) -> np.ndarray:
    """ROI time series (T x R) -> Fisher-z upper-triangle connectivity vector (R*(R-1)/2,)."""
    corr = np.nan_to_num(np.corrcoef(ts, rowvar=False))
    np.fill_diagonal(corr, 1.0)
    if method == "pearson":
        mat = corr
    elif method == "partial":
        mat = _precision_to_partial(LedoitWolf().fit(ts).precision_)
    elif method == "glasso":
        alpha = glasso_alpha
        for _ in range(6):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, prec = graphical_lasso(corr, alpha=alpha, max_iter=200)
                break
            except (FloatingPointError, ValueError):
                alpha *= 2
        else:
            raise RuntimeError("graphical lasso did not converge; increase cfg.glasso_alpha")
        mat = _precision_to_partial(prec)
    else:
        raise ValueError(method)
    z = np.arctanh(np.clip(mat, -0.999999, 0.999999))
    iu = np.triu_indices(mat.shape[0], k=1)
    return z[iu].astype(np.float32)


def _one_subject(path: Path, cfg: Config) -> np.ndarray:
    ts = clean_timeseries(_read_timeseries(path, cfg.fmri_roi_by_time), cfg)
    return connectivity_vector(ts, cfg.connectivity, cfg.glasso_alpha)


def load_fmri(cfg: Config):
    """Returns (ids, features[N x R(R-1)/2], feature_names). Cached on disk."""
    root = Path(cfg.fmri_dir)
    files = {}
    for p in sorted(root.iterdir()):
        if p.suffix.lower() in cfg.fmri_extensions:
            files[str(cfg.fmri_id_from_filename(p.stem)).strip()] = p
    if not files:
        raise FileNotFoundError(f"No fMRI files with extensions {cfg.fmri_extensions} in {root}")

    key = json.dumps([cfg.connectivity, cfg.glasso_alpha, cfg.fmri_tr, cfg.fmri_bandpass, cfg.fmri_roi_by_time,
                      sorted(files)], default=str)
    cache = Path(cfg.cache_dir) / f"fmri_{hashlib.md5(key.encode()).hexdigest()[:10]}.npz"
    ids = list(files)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        feats = z["feats"]
    else:
        print(f"[fmri] computing '{cfg.connectivity}' connectivity for {len(ids)} subjects ...")
        feats = np.stack(Parallel(n_jobs=cfg.n_jobs)(delayed(_one_subject)(files[i], cfg) for i in ids))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, feats=feats)

    # ROI count from vector length: F = R(R-1)/2
    n_feat = feats.shape[1]
    n_roi = int(round((1 + np.sqrt(1 + 8 * n_feat)) / 2))
    if n_roi * (n_roi - 1) // 2 != n_feat:
        raise ValueError("subjects have different numbers of ROIs")
    if cfg.roi_labels_file:
        labels = [l.strip() for l in open(cfg.roi_labels_file) if l.strip()]
        assert len(labels) == n_roi, f"roi_labels_file has {len(labels)} names but data has {n_roi} ROIs"
    else:
        labels = [f"ROI{i + 1}" for i in range(n_roi)]
    iu = np.triu_indices(n_roi, k=1)
    names = [f"{labels[i]}~{labels[j]}" for i, j in zip(*iu)]
    return ids, feats, names


# =====================================================================================================================
# Clinical
# =====================================================================================================================
def load_clinical(cfg: Config):
    df = pd.read_csv(cfg.clinical_csv)
    need = [cfg.clinical_id_col, cfg.clinical_label_col, *cfg.clinical_numeric_cols, *cfg.clinical_categorical_cols]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"clinical csv is missing columns {missing}; available: {list(df.columns)}")
    df[cfg.clinical_id_col] = df[cfg.clinical_id_col].astype(str).str.strip()
    if df[cfg.clinical_id_col].duplicated().any():
        n = int(df[cfg.clinical_id_col].duplicated().sum())
        print(f"[clinical] WARNING: {n} duplicated subject rows (multiple visits?) - keeping the first per subject.")
        df = df.drop_duplicates(cfg.clinical_id_col, keep="first")
    lab = df[cfg.clinical_label_col].astype(str).str.strip().map(cfg.label_map)
    keep = lab.notna()
    if (~keep).any():
        print(f"[clinical] dropping {(~keep).sum()} subjects whose label is not in cfg.label_map")
    df, lab = df[keep], lab[keep]
    df = df.set_index(cfg.clinical_id_col)
    y = lab.map({c: i for i, c in enumerate(CLASS_NAMES)}).astype(int)
    y.index = df.index
    return df[cfg.clinical_numeric_cols + cfg.clinical_categorical_cols], y


# =====================================================================================================================
# SNP
# =====================================================================================================================
def load_snp(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.snp_csv)
    df[cfg.snp_id_col] = df[cfg.snp_id_col].astype(str).str.strip()
    df = df.drop_duplicates(cfg.snp_id_col).set_index(cfg.snp_id_col)
    df = df.drop(columns=[c for c in cfg.snp_drop_cols if c in df.columns])
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace(cfg.snp_missing_codes, np.nan)
    df = df.where(df.isin([0, 1, 2]))                    # anything that is not a valid genotype -> missing
    return df.astype(np.float32)


def snp_sample_qc(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    miss = df.isna().mean(axis=1)
    bad = miss > cfg.snp_max_sample_missing
    if bad.any():
        print(f"[snp] sample QC: dropping {int(bad.sum())} subjects with >{cfg.snp_max_sample_missing:.0%} missing")
    return df[~bad]


def hwe_pvalues(g: pd.DataFrame) -> np.ndarray:
    """Chi-square (1 d.o.f.) Hardy-Weinberg test per SNP column."""
    a = g.to_numpy()
    n0, n1, n2 = [(a == k).sum(0).astype(float) for k in (0, 1, 2)]
    n = n0 + n1 + n2
    p = (2 * n2 + n1) / np.maximum(2 * n, 1)
    q = 1 - p
    exp = np.stack([n * q * q, 2 * n * p * q, n * p * p])
    obs = np.stack([n0, n1, n2])
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(exp > 0, (obs - exp) ** 2 / exp, 0).sum(0)
    return stats.chi2.sf(chi, 1)


def snp_variant_qc(df: pd.DataFrame, y_by_id: pd.Series, cfg: Config) -> pd.DataFrame:
    n0 = df.shape[1]
    df = df.loc[:, df.isna().mean() <= cfg.snp_max_snp_missing]
    p = df.mean() / 2.0
    df = df.loc[:, np.minimum(p, 1 - p) >= cfg.snp_min_maf]
    if cfg.snp_hwe_p is not None:
        ctrl = df.loc[y_by_id.reindex(df.index) == CLASS_NAMES.index("CN")]
        ref = ctrl if len(ctrl) >= 20 else df
        df = df.loc[:, hwe_pvalues(ref) >= cfg.snp_hwe_p]
    print(f"[snp] variant QC: {n0} -> {df.shape[1]} SNPs (missingness, MAF, HWE)")
    return df


# =====================================================================================================================
# Subject matching
# =====================================================================================================================
@dataclass
class RawData:
    ids: List[str]
    y: np.ndarray
    fmri: Optional[np.ndarray]
    fmri_names: Optional[List[str]]
    clinical: pd.DataFrame
    snp: pd.DataFrame


def load_raw(cfg: Config) -> RawData:
    cfg.validate()
    use_fmri = "fmri" in cfg.modalities
    clin, y = load_clinical(cfg)
    snp = snp_sample_qc(load_snp(cfg), cfg)
    f_ids = f_feats = f_names = None
    if use_fmri:
        f_ids, f_feats, f_names = load_fmri(cfg)

    common = [i for i in clin.index if i in snp.index and (not use_fmri or i in set(f_ids))]
    print(f"[match] clinical={len(clin)}  SNP={len(snp)}" + (f"  fMRI={len(f_ids)}" if use_fmri else "")
          + f"  ->  common subjects={len(common)}")
    if len(common) < 20:
        raise ValueError("Fewer than 20 subjects present in all chosen modalities - check the subject IDs.")
    fmri = None
    if use_fmri:
        fi = {s: k for k, s in enumerate(f_ids)}
        fmri = f_feats[[fi[s] for s in common]]
    clin = clin.loc[common]
    y = y.loc[common].to_numpy()
    snp = snp_variant_qc(snp.loc[common], pd.Series(y, index=common), cfg)
    counts = {CLASS_NAMES[k]: int((y == k).sum()) for k in range(len(CLASS_NAMES))}
    print(f"[match] class counts: {counts}")
    return RawData(common, y, fmri, f_names, clin, snp)


# =====================================================================================================================
# Split-dependent preprocessing 
# =====================================================================================================================
@dataclass
class Bundle:
    ids: List[str]
    y: np.ndarray
    idx: Dict[str, np.ndarray]                # "train" / "val" / "test"
    X: Dict[str, np.ndarray]                  # modality -> (N x d) float32, preprocessed + selected
    names: Dict[str, List[str]]               # modality -> feature names


def make_split(y: np.ndarray, cfg: Config, seed: int) -> Dict[str, np.ndarray]:
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, test_size=cfg.val_frac + cfg.test_frac, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=cfg.test_frac / (cfg.val_frac + cfg.test_frac),
                              stratify=y[rest], random_state=seed)
    return {"train": np.sort(tr), "val": np.sort(va), "test": np.sort(te)}


def _select(k: Optional[int], d: int):
    return SelectKBest(f_classif, k=min(k, d)) if k else "passthrough"


def build_bundle(raw: RawData, cfg: Config, seed: int) -> Bundle:
    idx = make_split(raw.y, cfg, seed)
    tr, ytr = idx["train"], raw.y[idx["train"]]
    X, names = {}, {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # constant-feature warnings from f_classif

        # --- fMRI: scale -> top-k connections (train only)
        if raw.fmri is not None:
            pipe = Pipeline([("sc", StandardScaler()), ("sel", _select(cfg.fmri_topk_features, raw.fmri.shape[1]))])
            pipe.fit(raw.fmri[tr], ytr)
            X["fmri"] = pipe.transform(raw.fmri).astype(np.float32)
            mask = pipe.named_steps["sel"].get_support() if cfg.fmri_topk_features else np.ones(raw.fmri.shape[1], bool)
            names["fmri"] = list(np.asarray(raw.fmri_names)[mask])

        # --- SNP: impute (most frequent genotype) -> top-k SNPs -> scale (train only)
        snp = raw.snp.to_numpy()
        pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("sel", _select(cfg.snp_topk_features, snp.shape[1])), ("sc", StandardScaler())])
        pipe.fit(snp[tr], ytr)
        X["snp"] = pipe.transform(snp).astype(np.float32)
        mask = pipe.named_steps["sel"].get_support() if cfg.snp_topk_features else np.ones(snp.shape[1], bool)
        names["snp"] = list(raw.snp.columns[mask])

        # --- Clinical: median-impute + scale numeric, mode-impute + one-hot categorical (train only)
        parts = []
        if cfg.clinical_numeric_cols:
            parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
                          cfg.clinical_numeric_cols))
        if cfg.clinical_categorical_cols:
            parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                           ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                          cfg.clinical_categorical_cols))
        ct = ColumnTransformer(parts)
        clin = raw.clinical.copy()
        for c in cfg.clinical_numeric_cols:
            clin[c] = pd.to_numeric(clin[c], errors="coerce")
        ct.fit(clin.iloc[tr])
        X["clinical"] = ct.transform(clin).astype(np.float32)
        names["clinical"] = [n.split("__", 1)[-1] for n in ct.get_feature_names_out()]

    print(f"[bundle seed={seed}] train/val/test = {len(idx['train'])}/{len(idx['val'])}/{len(idx['test'])}  "
          f"features: " + ", ".join(f"{m}={X[m].shape[1]}" for m in X))
    return Bundle(raw.ids, raw.y, idx, X, names)

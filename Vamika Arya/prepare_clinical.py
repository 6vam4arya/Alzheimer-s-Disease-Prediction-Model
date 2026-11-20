
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DX_NUM = {1: "CN", 2: "MCI", 3: "AD"}
DXCHANGE = {1: "CN", 2: "MCI", 3: "AD", 4: "MCI", 5: "AD", 6: "AD", 7: "CN", 8: "MCI", 9: "CN"}


def read_rda(data_dir: Path, name: str) -> pd.DataFrame:
    import pyreadr
    f = data_dir / f"{name}.rda"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found")
    res = pyreadr.read_r(str(f))
    df = res[name] if name in res else next(iter(res.values()))
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    return df


def norm_dx(v, numeric_map):
    """Turn a diagnosis value (1/2/3, 'CN', 'MCI', 'Dementia', factor label ...) into CN / MCI / AD / NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
        return np.nan
    try:
        return numeric_map.get(int(float(v)), np.nan)
    except (TypeError, ValueError):
        pass
    t = str(v).strip().lower()
    if t in ("cn", "nl", "normal", "cognitively normal", "stable: nl"):
        return "CN"
    if "mci" in t or "mild cognitive" in t:
        return "MCI"
    if "dementia" in t or t in ("ad", "alzheimer's disease", "alzheimers disease"):
        return "AD"
    return np.nan


def to_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="1970-01-01", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def add_ptid(df: pd.DataFrame, data_dir: Path, name: str) -> pd.DataFrame:
    if "PTID" in df.columns:
        return df
    if "RID" not in df.columns:
        raise KeyError(f"{name}: neither PTID nor RID column. Columns: {list(df.columns)}")
    ros = read_rda(data_dir, "ROSTER")[["RID", "PTID"]].drop_duplicates("RID")
    return df.merge(ros, on="RID", how="left")


def build_diagnosis(data_dir: Path) -> pd.DataFrame:
    d = add_ptid(read_rda(data_dir, "DXSUM"), data_dir, "DXSUM")
    print("DXSUM columns:", list(d.columns))
    dx = pd.Series(np.nan, index=d.index, dtype=object)
    for col, mp in (("DIAGNOSIS", DX_NUM), ("DXCURREN", DX_NUM), ("DXCHANGE", DXCHANGE)):
        if col in d.columns:
            dx = dx.where(dx.notna(), d[col].map(lambda v: norm_dx(v, mp)))
    if dx.notna().sum() == 0:
        raise KeyError("could not find a usable diagnosis column (looked for DIAGNOSIS, DXCURREN, DXCHANGE)")
    date_col = next((c for c in ("EXAMDATE", "USERDATE") if c in d.columns), None)
    d = d.assign(DX=dx, EXAMDATE=to_date(d[date_col]) if date_col else pd.NaT)
    d = d[d.DX.notna() & d.PTID.notna()].sort_values(["PTID", "EXAMDATE"])
    return d.groupby("PTID", as_index=False).first()[["PTID", "DX", "EXAMDATE"]]


def build_demographics(data_dir: Path) -> pd.DataFrame:
    p = add_ptid(read_rda(data_dir, "PTDEMOG"), data_dir, "PTDEMOG")
    print("PTDEMOG columns:", list(p.columns))
    keep = [c for c in ("PTGENDER", "PTEDUCAT", "PTDOB") if c in p.columns]
    p = p.sort_values("PTID")
    out = p.groupby("PTID", as_index=False)[keep].first()          # first non-null value per subject
    if "PTGENDER" in out:
        out["PTGENDER"] = out["PTGENDER"].map(lambda v: {1: "Male", 2: "Female"}.get(
            int(v) if str(v).replace(".0", "").isdigit() else v, v))
    if "PTEDUCAT" in out:
        out["PTEDUCAT"] = pd.to_numeric(out["PTEDUCAT"], errors="coerce")
    if "PTDOB" in out:
        out["PTDOB"] = to_date(out["PTDOB"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="folder holding DXSUM.rda, PTDEMOG.rda ...")
    ap.add_argument("--out", default="clinical.csv")
    ap.add_argument("--snp_csv", help="optional: report overlap with the SNP subjects")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)

    dx, demo = build_diagnosis(data_dir), build_demographics(data_dir)
    df = dx.merge(demo, on="PTID", how="left")
    if "PTDOB" in df:
        df["AGE"] = ((df["EXAMDATE"] - df["PTDOB"]).dt.days / 365.25).round(1)
    else:
        df["AGE"] = np.nan
    for c in ("AGE", "PTGENDER", "PTEDUCAT"):
        if c not in df or df[c].isna().all():
            print(f"WARNING: {c} could not be built - remove it from clinical_*_cols in config.py")
    df = df[["PTID", "DX", "AGE", "PTGENDER", "PTEDUCAT", "EXAMDATE"]]
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(df)} subjects")
    print("diagnosis counts:", df.DX.value_counts().to_dict())
    print("missing:", {c: int(df[c].isna().sum()) for c in ("AGE", "PTGENDER", "PTEDUCAT")})
    if a.snp_csv:
        snp = set(pd.read_csv(a.snp_csv, usecols=["IID"]).IID.astype(str))
        both = df[df.PTID.astype(str).isin(snp)]
        print(f"subjects with BOTH diagnosis and SNPs: {len(both)}   classes: {both.DX.value_counts().to_dict()}")


if __name__ == "__main__":
    main()

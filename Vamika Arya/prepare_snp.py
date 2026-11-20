import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Lead SNPs from large AD GWAS (CLU, PICALM, CR1, BIN1, ABCA7, CD33, MS4A, EPHA1, CD2AP, SORL1, ...).
# Only those present in your .bim (by rsID) are used; the script prints which ones were found.
AD_GWAS_SNPS = [
    "rs11136000", "rs9331896",    # CLU
    "rs3851179", "rs10792832",    # PICALM
    "rs6656401", "rs3818361",     # CR1
    "rs6733839", "rs744373",      # BIN1
    "rs3764650", "rs4147929",     # ABCA7
    "rs3865444",                  # CD33
    "rs610932", "rs983392",       # MS4A6A / MS4A
    "rs11771145", "rs11767557",   # EPHA1
    "rs9349407", "rs10948363",    # CD2AP
    "rs11218343",                 # SORL1
    "rs9271192",                  # HLA-DRB5/DRB1
    "rs28834970",                 # PTK2B
    "rs190982",                   # MEF2C
    "rs2718058",                  # NME8
    "rs1476679",                  # ZCWPW1
    "rs10838725",                 # CELF1
    "rs17125944",                 # FERMT2
    "rs7274581",                  # CASS4
    "rs10498633",                 # SLC24A4
]
APOE_SNPS = ["rs429358", "rs7412"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", required=True, help="path to the .bed file (.bim and .fam must sit next to it)")
    ap.add_argument("--out", default="snp_genotypes.csv")
    ap.add_argument("--snp_list", help="text file with extra rsIDs, one per line")
    ap.add_argument("--n_random", type=int, default=0, help="add N random common SNPs (MAF>=0.05, <2%% missing)")
    ap.add_argument("--include_apoe", action="store_true", help="also keep the two APOE SNPs (rs429358, rs7412)")
    ap.add_argument("--id_from", default="iid", choices=["iid", "fid"], help="which .fam column holds the subject ID")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from bed_reader import open_bed
    bed = open_bed(a.bed)
    ids = np.asarray(bed.iid if a.id_from == "iid" else bed.fid).astype(str)
    sid = pd.Index(np.asarray(bed.sid).astype(str))
    print(f"subjects: {len(ids)}   SNPs: {len(sid):,}")
    print(f"first subject IDs: {list(ids[:5])}   (ADNI PTID looks like 002_S_0295)")
    print(f"first SNP IDs:     {list(sid[:5])}")

    wanted = list(AD_GWAS_SNPS) + (APOE_SNPS if a.include_apoe else [])
    if a.snp_list:
        wanted += [l.strip() for l in open(a.snp_list) if l.strip()]
    wanted = list(dict.fromkeys(wanted))
    pos = sid.get_indexer(wanted)
    found = [(w, p) for w, p in zip(wanted, pos) if p >= 0]
    print(f"curated/custom SNPs found in file: {len(found)} / {len(wanted)}")
    print("  missing:", [w for w, p in zip(wanted, pos) if p < 0])
    cols = [p for _, p in found]

    if a.n_random > 0:
        rng = np.random.default_rng(a.seed)
        cand = rng.choice(len(sid), size=min(len(sid), a.n_random * 4), replace=False)
        cand = np.sort(np.setdiff1d(cand, cols))
        g = bed.read(index=np.s_[:, cand], dtype="float32")
        miss = np.isnan(g).mean(0)
        maf = np.nanmean(g, axis=0) / 2
        maf = np.minimum(maf, 1 - maf)
        ok = cand[(miss < 0.02) & (maf >= 0.05)][:a.n_random]
        print(f"random common SNPs kept: {len(ok)}")
        cols += list(ok)

    if not cols:
        raise SystemExit("No SNPs selected. Check that your .bim uses rsIDs, or pass --snp_list / --n_random.")
    cols = np.array(cols)
    g = bed.read(index=np.s_[:, cols], dtype="float32")             # NaN = missing genotype
    df = pd.DataFrame(g, columns=sid[cols])
    df.insert(0, "IID", ids)
    df = df.drop_duplicates("IID")
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {df.shape[0]} subjects x {df.shape[1] - 1} SNPs; "
          f"missing genotypes = {df.iloc[:, 1:].isna().mean().mean():.2%}")


if __name__ == "__main__":
    main()

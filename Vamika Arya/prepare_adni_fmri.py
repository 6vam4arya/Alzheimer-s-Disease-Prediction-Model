from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


# stage 1
def _dcm2niix_cmd() -> list:
   
    exe = shutil.which("dcm2niix")
    return [exe] if exe else [sys.executable, "-m", "dcm2niix"]


def find_series(dicom_root: Path) -> pd.DataFrame:
    """Walk ADNI/<PTID>/<series>/<date>/<image id>/ and list every folder that holds .dcm files."""
    rows = []
    for d in dicom_root.rglob("*"):
        if d.is_dir() and any(f.suffix.lower() == ".dcm" for f in d.iterdir()):
            parts = d.relative_to(dicom_root).parts
            if len(parts) >= 4:
                ptid, series, date, image_id = parts[0], parts[1], parts[2], parts[3]
                rows.append({"PTID": ptid, "series": series, "scan_date": date[:10], "image_id": image_id,
                             "dicom_dir": str(d)})
    if not rows:
        raise FileNotFoundError(f"no DICOM folders found under {dicom_root}")
    return pd.DataFrame(rows)


def convert(dicom_root: Path, work: Path, visit: str = "first") -> pd.DataFrame:
    series = find_series(dicom_root).sort_values(["PTID", "scan_date"])
    if visit == "first":                                    # one scan per subject: the earliest one
        series = series.groupby("PTID", as_index=False).first()
    nii_dir = work / "nifti"; nii_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in series.itertuples():
        out = nii_dir / f"{r.PTID}.nii.gz"
        js = nii_dir / f"{r.PTID}.json"
        if not out.exists():
            tmp = nii_dir / f"_{r.PTID}"; tmp.mkdir(exist_ok=True)
            proc = subprocess.run(_dcm2niix_cmd() + ["-z", "y", "-b", "y", "-f", "%i", "-o", str(tmp), r.dicom_dir],
                                  check=False, capture_output=True, text=True)
            cands = sorted(tmp.glob("*.nii.gz"), key=lambda p: p.stat().st_size, reverse=True)
            if not cands:
                print(f"[convert] FAILED {r.PTID}: {(proc.stdout + proc.stderr)[-500:]}"); continue
            cands[0].rename(out)
            j = cands[0].with_suffix("").with_suffix(".json")
            if j.exists():
                j.rename(js)
            for f in tmp.glob("*"):
                f.unlink()
            tmp.rmdir()
        meta = json.load(open(js)) if js.exists() else {}
        import nibabel as nib
        shape = nib.load(out).shape
        recs.append({"PTID": r.PTID, "scan_date": r.scan_date, "image_id": r.image_id, "series": r.series,
                     "nifti": str(out), "TR": meta.get("RepetitionTime"), "n_vols": shape[3] if len(shape) == 4 else 1})
        print(f"[convert] {r.PTID}  shape={shape}  TR={meta.get('RepetitionTime')}")
    man = pd.DataFrame(recs)
    man.to_csv(work / "manifest.csv", index=False)
    return man


#atlas
def load_atlas(name: str, atlas_file: str | None, atlas_labels: str | None):
    """Returns (nibabel label image with values 1..R, list of R ROI names). Default: AAL, first 90 (cerebrum) regions."""
    import nibabel as nib
    if atlas_file:
        img = nib.load(atlas_file)
        names = [l.strip() for l in open(atlas_labels)] if atlas_labels else None
        ids = [int(v) for v in np.unique(np.asarray(img.dataobj)) if v != 0]
        names = names or [f"ROI{i}" for i in ids]
    else:
        from nilearn import datasets
        a = datasets.fetch_atlas_aal(version="SPM12")
        img = nib.load(a.maps) if isinstance(a.maps, (str, Path)) else a.maps
        ids = [int(i) for i in a.indices]
        names = list(a.labels)
        keep = [k for k, i in enumerate(ids) if i < 7000][:90]          # 90 cerebral AAL regions (drop cerebellum/vermis)
        ids, names = [ids[k] for k in keep], [names[k] for k in keep]
    data = np.asarray(img.dataobj).astype(int)
    out = np.zeros_like(data)
    for new, old in enumerate(ids, start=1):
        out[data == old] = new
    return nib.Nifti1Image(out.astype(np.int16), img.affine), names


# stage 2
def _to_ants(nib_img, tmp: Path, name: str):
    """nibabel -> ANTs via a temp file (stable across ANTsPy versions)."""
    import ants, nibabel as nib
    f = tmp / f"{name}.nii.gz"; nib.save(nib_img, f)
    return ants.image_read(str(f))


def _from_ants(ants_img, tmp: Path, name: str):
    import ants, nibabel as nib
    f = tmp / f"{name}_out.nii.gz"; ants.image_write(ants_img, str(f))
    return nib.load(str(f))


def preprocess_subject(nifti: str, ptid: str, tr: float | None, atlas_img, out_dir: Path, work: Path,
                       drop: int = 5, max_vols: int | None = None, reg: str = "SyNRA") -> dict | None:
    import ants
    import nibabel as nib
    from nilearn import image as nimg, masking, signal
    from nilearn.datasets import load_mni152_template

    out_csv = out_dir / f"{ptid}.csv"
    if out_csv.exists():
        return None
    img = nib.load(nifti)
    if len(img.shape) != 4:
        print(f"[prep] {ptid}: not 4-D, skipped"); return None
    tr = tr or float(img.header.get_zooms()[3])
    data = img.slicer[..., drop:(drop + max_vols) if max_vols else None]
    n_t = data.shape[3]
    if n_t < 50:
        print(f"[prep] {ptid}: only {n_t} volumes after dropping, skipped"); return None

    # -- motion correction (rigid, to the mean volume)
    tmp = work / f"_tmp_{ptid}"; tmp.mkdir(parents=True, exist_ok=True)
    a4 = _to_ants(data, tmp, "bold")
    mc = ants.motion_correction(a4, type_of_transform="Rigid")
    mc_img = _from_ants(mc["motion_corrected"], tmp, "mc")
    fd = np.asarray(mc["FD"], dtype=float)
    params = np.stack([np.asarray(ants.read_transform(tx[0]).parameters, dtype=float)[:6]
                       for tx in mc["motion_parameters"]])
    mean_epi = nimg.mean_img(mc_img)
    brain = masking.compute_epi_mask(mean_epi)

    # -- mean EPI -> MNI, then bring the atlas into the subject's EPI space 
    tpl = load_mni152_template(resolution=2)
    fixed = _to_ants(tpl, tmp, "tpl")
    moving = _to_ants(nimg.math_img("a*m", a=mean_epi, m=brain), tmp, "mean_epi")
    moving = ants.n4_bias_field_correction(moving)
    r = ants.registration(fixed=fixed, moving=moving, type_of_transform=reg, aff_metric="mattes")
    atlas_mni = nimg.resample_to_img(atlas_img, tpl, interpolation="nearest")
    atlas_native = ants.apply_transforms(fixed=_to_ants(mean_epi, tmp, "mean_epi_fixed"), moving=_to_ants(atlas_mni, tmp, "atlas"),
                                         transformlist=r["invtransforms"], interpolator="nearestNeighbor")
    lab = np.rint(_from_ants(atlas_native, tmp, "atlas_native").get_fdata()).astype(int)

    # -- ROI mean time series 
    vol = mc_img.get_fdata(dtype=np.float32)
    bmask = brain.get_fdata() > 0
    n_roi = int(np.asarray(atlas_img.dataobj).max())
    ts = np.zeros((n_t, n_roi), dtype=np.float32)
    empty = []
    for k in range(1, n_roi + 1):
        m = (lab == k) & bmask
        if m.sum() < 3:
            empty.append(k); continue
        ts[:, k - 1] = vol[m].mean(axis=0)

    # -- nuisance regression + detrend + band-pass + z-score 
    deriv = np.vstack([np.zeros((1, params.shape[1])), np.diff(params, axis=0)])
    tcc = nimg.high_variance_confounds(mc_img, n_confounds=5, percentile=2.0, mask_img=brain)
    conf = np.column_stack([params, deriv, params ** 2, deriv ** 2, tcc])        # 24 motion + 5 tCompCor
    clean = signal.clean(ts, confounds=conf, detrend=True, standardize="zscore_sample",
                         low_pass=0.1 if tr < 5 else None, high_pass=0.01, t_r=tr)
    clean[:, [k - 1 for k in empty]] = 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, clean, delimiter=",", fmt="%.6f")
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    return {"PTID": ptid, "n_vols": n_t, "TR": tr, "mean_FD": float(np.nanmean(fd)), "max_FD": float(np.nanmax(fd)),
            "empty_rois": len(empty)}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip"); g.add_argument("--dicom_root")
    ap.add_argument("--work", default="work"); ap.add_argument("--out", default="fmri_roi_timeseries")
    ap.add_argument("--atlas", default="aal"); ap.add_argument("--atlas_file"); ap.add_argument("--atlas_labels")
    ap.add_argument("--drop", type=int, default=5, help="initial volumes to discard")
    ap.add_argument("--max_vols", type=int, default=None)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--visit", default="first", choices=["first"])
    ap.add_argument("--only", nargs="*", help="process just these PTIDs (for a trial run)")
    ap.add_argument("--max_fd", type=float, default=0.5, help="subjects with mean FD above this are flagged in qc.csv")
    a = ap.parse_args()

    work, out = Path(a.work), Path(a.out); work.mkdir(parents=True, exist_ok=True)
    root = Path(a.dicom_root) if a.dicom_root else work / "dicom"
    if a.zip:
        if not root.exists():
            print("extracting zip ..."); zipfile.ZipFile(a.zip).extractall(root)
        root = root / "ADNI" if (root / "ADNI").exists() else root
    man = convert(root, work, a.visit)
    if a.only:
        man = man[man.PTID.isin(a.only)]

    atlas_img, names = load_atlas(a.atlas, a.atlas_file, a.atlas_labels)
    out.mkdir(parents=True, exist_ok=True)
    (out.parent / "roi_labels.txt").write_text("\n".join(names))
    res = Parallel(n_jobs=a.n_jobs)(delayed(preprocess_subject)(r.nifti, r.PTID, r.TR, atlas_img, out, work, a.drop,
                                                              a.max_vols) for r in man.itertuples())
    qc = pd.DataFrame([r for r in res if r])
    if len(qc):
        qc["high_motion"] = qc.mean_FD > a.max_fd
        qc_path = out.parent / "qc.csv"
        qc.to_csv(qc_path, mode="a", header=not qc_path.exists(), index=False)
        print(qc.to_string(index=False))
    print(f"\nDone. Point config.fmri_dir at: {out.resolve()}   and config.roi_labels_file at: "
          f"{(out.parent / 'roi_labels.txt').resolve()}")


if __name__ == "__main__":
    main()

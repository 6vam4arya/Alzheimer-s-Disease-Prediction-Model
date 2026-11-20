from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

CLASS_NAMES = ["CN", "MCI", "AD"]                   # class index = position in this list (ADNI3/4 diagnoses)
MODALITIES = ("fmri", "clinical", "snp")


def fmri_id_from_filename(stem: str) -> str:
    return stem


@dataclass
class Config:
   
    # Any subset of ("fmri", "clinical", "snp").  Only subjects present in ALL chosen modalities are used.
    modalities: tuple = ("clinical", "snp")

   
    fmri_dir: str = "PATH/TO/fmri_roi_timeseries"
    fmri_extensions: tuple = (".npy", ".csv", ".tsv", ".txt")
    fmri_roi_by_time: bool = False                 
    roi_labels_file: Optional[str] = None         
    fmri_id_from_filename: Callable[[str], str] = fmri_id_from_filename

    # Clinical: ONE ROW PER SUBJECT 
    clinical_csv: str = "clinical.csv"          # written by prepare_clinical.py
    clinical_id_col: str = "PTID"
    clinical_label_col: str = "DX"
    # raw label value in your file -> one of CLASS_NAMES.  Rows with other values are dropped.
    label_map: Dict[str, str] = field(default_factory=lambda: {
        "CN": "CN", "MCI": "MCI", "AD": "AD",
    })
   
    
    clinical_numeric_cols: List[str] = field(default_factory=lambda: ["AGE", "PTEDUCAT"])
    clinical_categorical_cols: List[str] = field(default_factory=lambda: ["PTGENDER"])

    # SNP: ONE ROW PER SUBJECT, one column per SNP, values 0/1/2 (alt-allele count). PLINK `--recode A` (.raw) works
    #      too: set snp_id_col="IID" and keep snp_drop_cols.
    snp_csv: str = "snp_genotypes.csv"          # written by prepare_snp.py
    snp_id_col: str = "IID"
    snp_drop_cols: List[str] = field(default_factory=lambda: ["FID", "PAT", "MAT", "SEX", "PHENOTYPE"])
    snp_missing_codes: List[float] = field(default_factory=lambda: [-9])   # besides NaN / empty

    #  fMRI preprocessing
    connectivity: str = "glasso"       
    glasso_alpha: float = 0.05          
    fmri_tr: Optional[float] = None     # repetition time in sec; set together with bandpass to band-pass filter
    fmri_bandpass: Optional[tuple] = None   # e.g. (0.01, 0.1)
    fmri_topk_features: Optional[int] = 500     # keep top-k connections by ANOVA F on TRAIN only (None = all)
    n_jobs: int = 2

    # SNP quality control
    snp_max_sample_missing: float = 0.10    # drop subjects with > 10 % missing genotypes
    snp_max_snp_missing: float = 0.05       # drop SNPs with > 5 % missing
    snp_min_maf: float = 0.01               # drop rare SNPs
    snp_hwe_p: Optional[float] = 1e-6       # Hardy-Weinberg filter (computed in CN subjects); None = off
    snp_topk_features: Optional[int] = 200  # keep top-k SNPs by ANOVA F on TRAIN only (None = all)

    #  splits / general
    val_frac: float = 0.15
    test_frac: float = 0.15
    seeds: tuple = (0, 1, 2, 3, 4)
    out_dir: str = "results"
    cache_dir: str = "cache"

    # models
    latent_dim: int = 32                # 13,698 -> 32-D in your notes (per modality here)
    ae_hidden: int = 256
    ae_epochs: int = 100
    ae_lr: float = 1e-3

    graph_k: int = 10                   # top-K neighbours per patient (cosine)
    graph_weighted: bool = True         # edge weight = cosine similarity (else binary)

    dropout: float = 0.3
    weight_decay: float = 5e-4
    use_class_weights: bool = True
    select_metric: str = "macro_f1"     # model selection on validation: "macro_f1" | "balanced_acc" | "acc"

    # MLP
    mlp_hidden: List[int] = field(default_factory=lambda: [256, 64])
    mlp_lr: float = 1e-3
    mlp_epochs: int = 100
    batch_size: int = 32

    # GCN
    gcn_hidden: int = 64
    gcn_lr: float = 5e-3
    gcn_epochs: int = 100               
    # GAN (semi-supervised GAN: discriminator = (K)-class classifier + real/fake)
    gan_z_dim: int = 64
    gan_hidden: int = 256
    gan_epochs: int = 200
    gan_lr: float = 2e-4
    gan_unsup_weight: float = 1.0
    gan_use_heldout_unlabeled: bool = False   # True lets D see val/test FEATURES (never labels) -> transductive

    # XAI (integrated gradients)
    ig_steps: int = 50
    ig_max_subjects: int = 50

    def validate(self):
        import os
        need = {"fmri": "fmri_dir", "clinical": "clinical_csv", "snp": "snp_csv"}
        if "clinical" not in self.modalities:
            raise ValueError("'clinical' must be in cfg.modalities (it holds the diagnosis labels)")
        for m in self.modalities:
            name = need[m]
            p = getattr(self, name)
            if p.startswith("PATH/TO") or not os.path.exists(p):
                raise FileNotFoundError(
                    f"config.{name} = '{p}' does not exist. Edit the 'YOUR DATA' section of config.py.")
        if self.connectivity not in ("pearson", "partial", "glasso"):
            raise ValueError("connectivity must be pearson | partial | glasso")

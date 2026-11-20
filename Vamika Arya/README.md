# Multimodal Alzheimer's detection: MLP vs GCN vs GAN (fMRI BOLD + clinical + SNP)

Implements the pipeline from your notes. Classes: CN, EMCI, LMCI, AD.

## Setup
```
pip install -r requirements.txt
```
Edit the **YOUR DATA** block of `config.py` (paths, ID/label columns, label map, clinical columns), then:
```
python run_experiments.py --explain
python run_experiments.py --models gcn --sets multimodal --seeds 0 1 2     # subsets
```

## Expected data
| Modality | Format |
|---|---|
| fMRI | one file per subject (`.npy/.csv/.tsv/.txt`), shape *timepoints x ROIs*, already denoised (motion, physiology, non-neuronal signal). File stem = subject ID (edit `fmri_id_from_filename`). |
| Clinical | one row per subject; ID column, label column, numeric + categorical feature columns |
| SNP | one row per subject; ID column + one column per SNP with 0/1/2 (PLINK `--recode A` `.raw` works) |

Only subjects present in **all three** files are used, so imaging/clinical/SNP-only comparisons use the same people.

## Pipeline
1. **fMRI**: detrend, optional band-pass, z-score -> `glasso` (graphical lasso -> partial correlation, as in notes) / `partial` / `pearson` -> Fisher-z -> upper-triangle vector. Cached in `cache/`.
2. **SNP**: sample missingness, SNP missingness, MAF, Hardy-Weinberg (in CN) QC -> mode imputation -> top-k by ANOVA F -> standardise.
3. **Clinical**: median/mode imputation, standardise, one-hot.
4. **Split** (stratified train/val/test per seed). Imputers, scalers and top-k selection are **fit on train only**.
5. **MLP**: concatenated features -> MLP (BatchNorm, dropout 0.3).
6. **GCN**: per-modality autoencoder (-> 32-D, trained on train subjects) -> unified patient embedding -> pairwise cosine, top-K graph -> 2 GCN layers -> dense head (BN + dropout 0.3) -> 4 classes. Adam + cross-entropy, 100 epochs, checkpoint with best validation score. Graph uses features only; the loss uses train labels only.
7. **GAN**: semi-supervised GAN. The discriminator is the 4-class classifier (plus real/fake); the generator is trained with feature matching. Generated samples are never added to any dataset and never used for evaluation.
8. **XAI** (`--explain`): integrated gradients on test AD subjects -> `results/ig_*.csv` (per-feature importance: SNPs, ROI~ROI connections, clinical factors, plus modality share).
9. **Final comparison**: `results/summary.csv` + printed delta of multimodal vs best single modality.

## Things to watch
- **Clinical leakage**: keep MMSE/CDR/ADAS-type scores out of `clinical_numeric_cols` unless intended; they nearly define the diagnosis.
- **Small N** (e.g. 240 subjects): test sets are tiny, so use several seeds and report mean ± std; differences of a few points are noise.
- The graph is transductive (val/test patients are nodes, with features but no labels). Set `gan_use_heldout_unlabeled=False` (default) to keep the GAN inductive.
- If you use ADNI, make sure the same subject is not in train and test under different IDs/visits.
- Hyper-parameters (`graph_k`, `glasso_alpha`, `*_topk_features`, learning rates) are starting points: tune on validation only.

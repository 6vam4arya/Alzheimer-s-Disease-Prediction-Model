# Multimodal Alzheimer's Disease Classification (ADNI)

A research notebook that asks one question: **if we give a neural network three different views of the same person (their brain activity, their DNA and their cognitive test scores), how well can it tell cognitively normal people, people with mild cognitive impairment, and people with Alzheimer's disease apart?**

Everything lives in a single notebook, `computervisionalz-final.ipynb`. It takes raw ADNI files and produces a matched cohort, trained models, comparison experiments and figures.

> **Research use only.** This is not a diagnostic tool.

---

## The project in one picture

<img width="2520" height="1812" alt="image" src="https://github.com/user-attachments/assets/f2b601f0-3c5b-48ff-b3ab-9130c7f65af9" />


The top strip is the data journey: raw files are aligned, filtered, split and cleaned. The two panels underneath are the two network designs that consume the result.

---

## The three views of a participant

| View | What it captures | What the notebook uses |
|---|---|---|
| **fMRI** | How strongly different brain regions activate together | Precomputed connectivity measures from a CSV (no raw brain images are processed) |
| **Genetics (SNP)** | Inherited variation at specific points of the genome | Genotypes decoded from PLINK `.bed/.bim/.fam` files |
| **Clinical / cognitive** | Age, sex, education and scores for memory, executive function, language and visuospatial ability | ADNI `.rda` tables, converted to CSV through R |

Each view has a different shape, scale and meaning, so the models never mix them in their first layer. Every view gets its own small network first.

---

## How the pipeline works

**1. Load the files.** The notebook reads the fMRI table and the PLINK metadata. It calls R (`Rscript`) to convert the clinical and cognitive `.rda` files into CSVs that Python can read.

**2. Link people across datasets.** The fMRI table and the genetic files use different identifiers. The notebook matches them through the participant ID (`RID`), so each person can be followed across all three sources.

**3. Match scans to clinic visits.** One participant can have many scans and many clinic visits. A scan is only paired with a clinical and a cognitive record when their dates are close, with a maximum gap of 90 days. This keeps the three views describing the same moment in time. Duplicate visit records are resolved using dates.

**4. Filter the cohort.** The notebook keeps scans that have a diagnosis of CN, MCI or AD, a nearby clinical and cognitive record, and a genetic match. It then removes duplicate participant-visit rows. A cohort-flow table records how many scans survive each step.

**5. Build features per view.**
- *fMRI:* keep connectivity-style columns, convert to numbers, and drop empty or mirrored columns.
- *SNP:* decode genotypes, apply quality control, keep the most variable candidates, prune variants that are highly correlated with their neighbours (LD pruning), and keep a pool of up to 5,000. Later, within each training split, a statistical test picks the final 50.
- *Clinical:* use age and demographic fields, the cognitive domain scores and their standard errors, and derived summaries (mean, minimum and spread across domains, plus memory-minus-executive and memory-minus-language differences).

**6. Split by person, not by scan.** Participants are divided 70/15/15 into training, validation and test groups. All of a person's scans stay in one group, so the model is never tested on someone it has already seen. This is repeated for five random seeds.

**7. Fit preprocessing on training data only.** Missing-value filling, scaling and SNP selection learn their settings from the training group and are then applied unchanged to the validation and test groups. Nothing about the test data leaks into preparation.

**8. Train, compare, interpret.** The models are trained, evaluated across seeds, and compared through the ablations listed below.

---

## The two network designs

### A. Three-branch MLP (the straightforward design)

An MLP (multilayer perceptron) is a stack of fully connected layers. Here there are three small stacks, one per view, followed by a shared stack.

```
fMRI input  ─► Dense 256 ─► BatchNorm ─► Dropout 0.30 ─► Dense 128 ─► Dense 64 ─┐
SNP input   ─► Dense 128 ─► BatchNorm ─► Dropout 0.30 ─► Dense 64  ─► Dense 32 ─┼─► Concatenate (128)
Clinical    ─► Dense 64  ─► BatchNorm ─► Dropout 0.20 ─► Dense 32               ─┘          │
                                                                                            ▼
                              Dense 128 ─► BatchNorm ─► Dropout 0.40 ─► Dense 64 ─► Dropout 0.25 ─► Dense 3 (softmax)
```

- **Why separate branches?** Each view is translated into its own compact summary before anything is combined.
- **Why concatenate?** Joining the three summaries into one 128-number vector is the simplest way to let later layers look at all views together (feature-level fusion).
- **What do BatchNorm and Dropout do?** BatchNorm keeps the numbers flowing through the layers on a stable scale. Dropout randomly switches off some units during training so the network cannot lean too heavily on any single one.
- **The output** is three probabilities, one each for CN, MCI and AD.

### B. HAMF-style hierarchical attention model (the "weigh the views" design)

This design is inspired by the hierarchical attention multimodal fusion (HAMF) framework from the literature. It is an adaptation, not an exact reproduction.

Instead of simply gluing the summaries together, it lets the network learn how much each view should count.

1. **Embed.** Each view passes through Dense 128, BatchNorm and Dropout, then a linear layer that produces a 64-number embedding.
2. **Gate.** A sigmoid gate (values between 0 and 1) is multiplied element by element with the embedding. This lets the network dim or keep individual dimensions.
3. **Pairwise attention.** For each pair of views (fMRI + SNP, fMRI + Clinical, Clinical + SNP) a small keyless attention layer scores the two vectors, converts the scores to weights with a softmax, and returns the weighted sum.
4. **Second-level attention.** The three pairwise outputs are combined by the same attention mechanism into one fused vector.
5. **Classify.** Dense 32, Dropout 0.25, then a 3-way softmax.

When only two views are present, a single attention step is used. With one view, there is nothing to fuse.

**Keyless attention in one line:** every input vector gets a learned score `e = wᵀx`, the scores are turned into weights `λ = softmax(e)`, and the output is `Σ λ·x`.

---

## Training recipe

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 0.001 |
| Batch size | 32 |
| Maximum epochs | 100 |
| Early stopping | Watch validation loss, patience 10, restore best weights |
| Class weights | Computed from training labels, capped at 3.0 |
| Seeds | 42, 7, 2024, 11, 99 |

Individual experiment cells may override these settings.

---

## Experiments included

| Experiment | What it varies |
|---|---|
| **Modality ablation** | All 7 non-empty subsets of the three views, for both network designs |
| **Fusion comparison** | Concatenation, general attention, hierarchical attention with linear gating, hierarchical attention with nonlinear gating |
| **Improved MLP** | Smaller layers, L2 regularization, modality dropout, optional auxiliary heads, seed ensembles |
| **Alternative SNP features** | APOE-related allele counts and selected Alzheimer's risk loci (where present in the BIM file) and ancestry principal components |
| **Classical baselines** | Logistic Regression and Gradient Boosting on the same features |
| **Interpretability** | SHAP feature attributions for the full model, where the analysis completes |

The improvement experiment runs in a fast, fixed-configuration mode (`QUICK=True`) with participant-grouped stratified folds. It is a sanity pass, not an exhaustive hyperparameter search.

---

## Metrics reported

- **Accuracy:** share of correct predictions.
- **Balanced accuracy:** the average of per-class recall, so small classes count equally.
- **Macro-F1:** F1 per class, averaged with equal weight.
- **ROC-AUC:** one-vs-rest, macro-averaged, from predicted probabilities.
- **Confusion matrices and classification reports:** to see which classes get mixed up.

---

## Outputs the notebook produces

The notebook writes everything to `/kaggle/working/outputs/` and bundles it into a ZIP:
- the cohort-flow table
- per-experiment metric tables (mean and standard deviation across seeds)
- confusion matrices
- the modality-ablation bar chart
- the fusion-comparison table
- the SHAP importance plot
- the improved-model and baseline comparison table

---

## Glossary

| Term | Meaning |
|---|---|
| **ADNI** | Alzheimer's Disease Neuroimaging Initiative, the source of the data |
| **CN / MCI / AD** | Cognitively normal / mild cognitive impairment / Alzheimer's disease |
| **fMRI connectivity** | A measure of how in-sync the activity of two brain regions is |
| **SNP** | Single-nucleotide polymorphism, a one-letter variation in DNA |
| **PLINK** | A standard file format and toolkit for genotype data |
| **LD pruning** | Dropping SNPs that are near-copies of their neighbours |
| **RID** | ADNI's numeric participant identifier |
| **Feature-level fusion** | Combining the learned summaries of each view before classifying |
| **Attention** | A learned weighting that decides how much each input contributes |
| **Gating** | A learned 0-to-1 dial applied to each value in a vector |
| **Data leakage** | Information from test participants accidentally influencing training |
| **SHAP** | A method for estimating how much each input feature moved a prediction |

---

## Design decisions at a glance

- **Split by participant:** repeated scans of one person never straddle train and test.
- **Fit on training data only:** imputation, scaling and SNP selection never see validation or test data.
- **Dated matching rule:** a 90-day window keeps the three views contemporaneous.
- **Separate branches:** views with different scales and meanings are encoded independently before fusion.
- **Same protocol everywhere:** every model and ablation uses the same splits and metrics, so comparisons are like for like.
- **Honest labelling:** the HAMF-style model is described as inspired by, not identical to, the published framework.

---

## Limitations and responsible use

- Research-only; not clinically validated.
- ADNI data are not redistributed here and are subject to ADNI's data-use terms.
- Cohort size and class balance affect how stable any metric is.
- Some cognitive scores overlap with how diagnoses are made, so clinical-containing models should be interpreted with that in mind.
- Matching is cross-sectional (scan and visit close in time); this is not longitudinal forecasting.
- fMRI inputs are precomputed features, not raw volumes.
- SHAP attributions describe the model, not biological causes.

## Acknowledgements

Data used in this project were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) and are subject to its access and data-use conditions.

# MILK10k lesion classification — Computer Vision and Speech Recognition

Individual course project (EADA). The repository grows one session at a time:
Session 1 was the setup and first EDA, Session 2 the extended EDA and a first
data pipeline, Session 3 / Milestone 1 turns that into a **leak-free,
reproducible pipeline**.

## 1. What this project is

**Task.** Given the two photos of a skin lesion — one dermoscopic (through a
dermatoscope) and one clinical close-up — predict its diagnosis.

- **Primary target:** `diagnosis_1` — Benign / Malignant / Indeterminate.
- **Stretch target:** the 11-class scheme in `training_gt.csv` (BCC, NV, BKL,
  SCCKA, MEL, AKIEC, DF, INF, VASC, BEN_OTH, MAL_OTH).

The prediction supports a doctor at triage, deciding whether a lesion needs a
biopsy. Missing a malignant lesion costs far more than an unnecessary check-up,
so the model is judged on macro-F1 and per-class recall, not accuracy.

**Dataset.** MILK10k — https://api.isic-archive.com/doi/milk10k/ (described in
*J Invest Dermatol*, doi 10.1016/j.jid.2025.06.1594), licence CC BY-NC,
attribution "MILK study team". 10,480 images / 5,240 lesions, exactly two images
per lesion.

**Goal of Milestone 1.** No model yet: a pipeline that loads and verifies the
data, decides the labels, splits it without leakage, preprocesses and augments it
defensibly, and feeds it in batches — all reproducible from one seed.

## 2. Setup and how to run

Python 3.11+. From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Where the data goes.** The dataset is downloaded to `Session1/milk10k/` by
`Session1/session1.py` (~345 MB). The path is defined in exactly one place,
`common/config.py`. To keep the images elsewhere, set an environment variable —
no path is ever hard-coded in a script or a notebook:

```bash
export MILK10K_DIR="/path/to/milk10k"
```

**Commands.**

```bash
python Session1/session1.py         # download the data + Session 1 EDA figures
python Session1/practice_ex1.py     # Session 1 exercises
python Session2/run_hw2.py          # Session 2 homework (EDA, processing, loader, visualiser)
python scripts/run_milestone1.py    # Milestone 1: B1–B8, writes everything in outputs/
jupyter notebook Session3/homework_part_a.ipynb    # Part A (or open it in VS Code)
```

`scripts/run_milestone1.py --quick` skips opening every image (the slow check in
B1). The notebook runs top to bottom with "Restart & Run All".

## 3. Repository structure

```
.
├── README.md                     this file
├── SUBMISSION.md                 link to every deliverable
├── requirements.txt              dependencies
├── .gitignore                    keeps images, venv and caches out of git
├── common/                       code shared by every session
│   ├── config.py                 paths, target column, seed, image size — the ONLY place paths are set
│   ├── milk10k.py                load_metadata(), available_subset()
│   └── viz.py                    show_grid(), plot_class_balance(), plot_batch_check()
├── src/                          the Milestone 1 pipeline (importable, no notebook code)
│   ├── lesions.py                per-image metadata -> one row per lesion
│   ├── splits.py                 split_lesions(), split checks, fold-strategy comparison
│   ├── labels.py                 label map, class weights, sampler weights
│   ├── quality.py                label cross-checks, integrity check, data-quality report
│   ├── transforms.py             train_transform / eval_transform + why each augmentation is safe
│   └── datasets.py               MilkImageDataset, LesionDataset, DataLoaders, aggregate_predictions
├── scripts/
│   └── run_milestone1.py         runs B1–B8 end to end
├── Session3/
│   └── homework_part_a.ipynb     Part A exercises (imports src/, contains no pipeline code)
├── outputs/                      everything generated (committed: small and it is the evidence)
│   ├── splits/{train,val,test}.csv
│   ├── lesions.csv, label_map.json, class_weights.json
│   ├── image_size_summary.csv, data_quality_report.md, augmentations.csv
│   └── figures/
├── Session1/                     session 1 scripts + figures (milk10k/ lives here, not committed)
└── Session2/                     session 2 homework + figures + results
```

**Why this layout.** Reusable logic lives in `common/` and `src/` as importable
modules, so the notebook and the scripts call the same functions instead of
copying them — when a split rule changes, it changes in one file. Notebooks are
for exploration and answers only. Everything a run produces goes to `outputs/`,
so generated files never mix with source code, and one config file holds the
paths, the seed and the image size.

## 4. Data handling rules

- **Not committed:** the images and the zip (`**/milk10k/`, `**/milk10k.zip`),
  the virtual environment, `__pycache__`. Anyone can recreate the data by running
  `Session1/session1.py`.
- **Committed:** all code, the split CSVs, the label map, the class weights, the
  quality report and the figures — they are small and they are the evidence for
  the reports.
- **Seed:** `config.SEED = 0`, used for the splits, the sampler and torch.
- **Splits created:** 29 September 2026, by `scripts/run_milestone1.py`, written
  to `outputs/splits/`.

## 5. Key decisions so far

**Label strategy.** `diagnosis_1` keeps all three classes: Indeterminate is only
2.3 % of lesions, but it means "the doctor could not decide", which is exactly the
case a tool must not quietly call benign. For the 11-class stretch goal the five
rare classes (DF, INF, VASC, BEN_OTH, MAL_OTH) are kept rather than merged —
merging would put benign and malignant lesions under one label — and handled with
class weights and a weighted sampler. The mapping is in `outputs/label_map.json`.

**Split design.** Splitting is done at **lesion level** with
`StratifiedGroupKFold` (70 / 15 / 15), then the images are attached. A naive
image-level split puts one photo of a lesion in train and the other in test: in
this dataset that leaks about 89 % of test lesions. Checks run on every split: no
lesion in two splits, exactly 2 images per lesion, class proportions per split
against the global mix.

**Preprocessing.** 224×224 (all images are 600×450, so every image is downscaled,
never upscaled), ImageNet normalisation. Both views of a lesion are used as
separate training samples with the lesion label; at evaluation the two views are
averaged back into **one prediction per lesion**
(`datasets.aggregate_predictions`), because the lesion is what a doctor decides
about.

**Augmentation.** Flips, small rotations and crops (scale ≥ 0.85, so the lesion
cannot be cropped out), plus mild brightness and contrast jitter. **Hue jitter is
set to 0**: A3.5 in the notebook measures that a hue shift moves colour more than
the gap between Benign and Malignant, so it would destroy the signal it is meant
to make robust. `eval_transform` is deterministic and is asserted to be so.

**Imbalance.** Measured on the train split only: 69 % of lesions are Malignant,
and in the 11-class scheme BCC outnumbers MAL_OTH by about 280 to 1. Handled with
both class weights (`outputs/class_weights.json`) and a `WeightedRandomSampler`.
Because MILK10k is biopsy-enriched, any accuracy measured here overstates
real-world performance — details in the Milestone 1 report.
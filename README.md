# MILK10k lesion classification — Computer Vision and Speech Recognition

Individual course project (EADA). The repository grows one session at a time:
Session 1 the setup and first EDA, Session 2 the extended EDA and a first data
pipeline, Session 3 / Milestone 1 a **leak-free, reproducible pipeline**,
Session 4 classical filters, Session 5 a classical ML baseline, Session 6 a
neural network on handcrafted features, Session 7 CNNs and transfer learning.
The scoreboard in section 6 is how every model is compared.

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
python Session4/session4.py         # Session 4: filters, Gaussian blur, Canny vs Sobel
python Session5/session5.py         # Session 5: classical ML baseline (features -> models -> test)
jupyter notebook Session3/homework_part_a.ipynb   # Part A (or open it in VS Code)
jupyter notebook Session6/Class6homework.ipynb    # Session 6: the MLP
jupyter notebook Session7/Class7.ipynb            # Session 7: CNNs and ResNet-18
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
│   ├── homework_part_a.ipynb     Part A exercises (imports src/, contains no pipeline code)
│   └── homework_part_a.pdf       the notebook exported to PDF
├── outputs/                      everything generated (committed: small and it is the evidence)
│   ├── splits/{train,val,test}.csv
│   ├── lesions.csv, label_map.json, class_weights.json
│   ├── image_size_summary.csv, data_quality_report.md, augmentations.csv
│   └── figures/
├── Session1/                     session 1 scripts + figures (milk10k/ lives here, not committed)
├── Session2/                     session 2 homework + figures + results
├── Session4/                     session 4 homework, self-contained
│   ├── filters.py                filter bank, Gaussian blur, Sobel, Canny (written by hand)
│   ├── session4.py               the three exercises
│   └── results.txt, figures/
├── Session5/                     session 5: the classical ML baseline
│   ├── features.py               image -> 40 numbers (colour, histograms, texture, GLCM, asymmetry)
│   ├── models.py                 the model zoo, the scaling pipeline and the scoring
│   ├── session5.py               crop -> feature set -> settings -> views -> test once
│   ├── features.csv              the cached features (committed: it is the input to Sessions 6-7)
│   └── results.txt, results/, figures/
├── Session6/                     session 6: a neural network (MLP) in PyTorch
│   ├── data.py                   loads the 33 features and the raw 32x32 pixels
│   └── Class6homework.ipynb      class work + the three homework questions
└── Session7/                     session 7: CNNs and transfer learning
    └── Class7.ipynb              class work + the mini challenge + the homework
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

## 6. Results so far — the scoreboard

Every number is **macro-F1**, not accuracy: about 72 % of the test set is
Malignant, so a model that always answers "Malignant" scores 0.717 accuracy while
learning nothing. The same model scores 0.278 macro-F1, and that is the floor.

All models use the same lesion-level split from Session 3 and the dermoscopic
photo of each lesion, so the rows are comparable. Validation chooses; test is
scored once.

| Session | Model | Validation | Test |
|---|---|---|---|
| 5 | majority (always Malignant) | 0.272 | 0.278 |
| 5 | random forest, 24 histogram features | 0.511 | 0.525 |
| 6 | MLP 33-64-3, handcrafted features | 0.463 | 0.483 |
| 6 | MLP 3072-64-3, raw 32x32 pixels | 0.524 | 0.456 |
| 7 | CNN from scratch | 0.520 | — |
| 7 | ResNet-18 frozen + linear head | 0.563 | — |
| 7 | ResNet-18, layer4 fine-tuned | 0.638 | — |
| 7 | **ResNet-18, layer3 + layer4 fine-tuned** | **0.642** | **0.590** |
| 7 | ResNet-18, all layers fine-tuned | 0.529 | — |

**Best model: ResNet-18 with `layer3`, `layer4` and the new head fine-tuned —
validation 0.642, test 0.590.** That is 2.1x the majority floor and 0.065 above
the Session 5 forest on test.

Three things the table shows:

- **Transfer learning is what moved the number.** A CNN trained from scratch on
  3,667 lesions reaches 0.520, barely past the Session 5 forest. The same
  architecture pretrained on ImageNet and partly fine-tuned reaches 0.642.
- **Fine-tuning everything made it worse** (0.529 vs 0.642). 11.7 million weights
  on 3,667 lesions is too much freedom; the early layers already hold the general
  edges and textures that transfer fine, so letting them move only loses them.
- **`layer3 + layer4` barely beat `layer4` alone** (0.642 vs 0.638) for about 2
  million extra trainable weights. On 786 validation lesions that gap is noise.

The drop from validation to test (0.642 -> 0.590) is the usual cost of having
chosen the configuration on validation, and is the honest number to quote.

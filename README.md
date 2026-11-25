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

**Where the data goes.** The dataset is downloaded to `data/milk10k/` by
`m1_data/Session1/session1.py` (~345 MB). The path is defined in exactly one place,
`common/config.py`. To keep the images elsewhere (Colab, for instance), set an
environment variable — no path is ever hard-coded in a script or a notebook:

```bash
export MILK10K_DIR="/path/to/milk10k"
```

**Commands.** Run from the repository root.

```bash
python m1_data/Session1/session1.py        # download the data + Session 1 EDA figures
python m1_data/Session1/practice_ex1.py    # Session 1 exercises
python m1_data/Session2/run_hw2.py         # Session 2 homework (EDA, processing, loader, visualiser)
python m1_data/run_milestone1.py           # Milestone 1: B1–B8, writes everything in outputs/
python m2_baseline/Session4/session4.py    # Session 4: filters, Gaussian blur, Canny vs Sobel
python m2_baseline/Session5/session5.py    # Session 5: classical ML baseline (features -> models -> test)
jupyter notebook m1_data/Session3/homework_part_a.ipynb   # Part A (or open it in VS Code)
jupyter notebook m3_deep/Session6/session6.ipynb          # Session 6: the MLP
jupyter notebook m3_deep/Session7/session7.ipynb          # Session 7: CNNs and ResNet-18
```

`m1_data/run_milestone1.py --quick` skips opening every image (the slow check in
B1). The notebooks run top to bottom with "Restart & Run All".

## 3. Repository structure

```
.
├── README.md                     this file
├── requirements.txt              dependencies
├── .gitignore                    keeps images, venv and caches out of git
├── data/milk10k/                 the dataset (downloaded, NOT committed)
│
├── common/                       SHARED - code every milestone uses
│   ├── config.py                 paths, target, seed, image size, the 3 and 11 class lists
│   ├── milk10k.py                load_metadata(), available_subset()
│   └── viz.py                    show_grid(), plot_class_balance(), plot_batch_check()
├── src/                          SHARED - the Milestone 1 pipeline
│   ├── lesions.py                per-image metadata -> one row per lesion
│   ├── splits.py                 split_lesions(), split checks, fold-strategy comparison
│   ├── labels.py                 label map, class weights, sampler weights
│   ├── quality.py                label cross-checks, integrity check, data-quality report
│   ├── transforms.py             train_transform / eval_transform + why each augmentation is safe
│   └── datasets.py               MilkImageDataset, LesionDataset, DataLoaders
├── outputs/                      SHARED - the splits every model is trained and scored on
│   ├── splits/{train,val,test}.csv
│   ├── lesions.csv, label_map.json, class_weights.json
│   ├── image_size_summary.csv, data_quality_report.md, augmentations.csv
│   └── figures/
│
├── m1_data/                      MILESTONE 1 - data, EDA, the pipeline
│   ├── run_milestone1.py         runs B1–B8 end to end
│   ├── Session1/                 download + first EDA
│   ├── Session2/                 homework 2: metadata, colour, processing, loader
│   └── Session3/                 homework part A (notebook + PDF)
├── m2_baseline/                  MILESTONE 2 - classical baseline
│   ├── Session4/                 filters, Gaussian blur, Canny vs Sobel (written by hand)
│   └── Session5/                 features -> random forest; features.csv is reused by Session 6
├── m3_deep/                      MILESTONE 3 - deep learning
│   ├── Session6/                 MLP in PyTorch (data.py + session6.ipynb)
│   └── Session7/                 CNN from scratch + ResNet-18 transfer learning (session7.ipynb)
├── m4_best_model/                MILESTONE 4 - improve the best model, error analysis
└── m5_final/                     MILESTONE 5 - predict.py, submission.csv, model card
```

**Why this layout.** Anything more than one milestone uses lives at the top:
`common/`, `src/`, `outputs/` and `data/`. The splits in `outputs/` are used by
M2, M3, M4 and M5 alike, so they can't belong to any single milestone. Each
`mX_` folder holds one stage's work, in the order the course built it. Every
script finds the repository by walking up until it sees `common/config.py`, so
nothing depends on how deep a file sits.

## 4. Data handling rules

- **Not committed:** the images and the zip (`**/milk10k/`, `**/milk10k.zip`),
  the virtual environment, `__pycache__`. Anyone can recreate the data by running
  `m1_data/Session1/session1.py`.
- **Committed:** all code, the split CSVs, the label map, the class weights, the
  quality report and the figures — they are small and they are the evidence for
  the reports.
- **Seed:** `config.SEED = 0`, used for the splits, the sampler and torch.
- **Splits created:** 29 September 2026, by `m1_data/run_milestone1.py`, written
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

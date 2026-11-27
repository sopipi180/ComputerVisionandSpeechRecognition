# Model card — MILK10k skin lesion classifier (11 classes)

Individual course project, EADA Computer Vision 2026 · Anne Sophie Christensen · leaderboard name `EADA-CV26-SC`

## At a glance

| | |
|---|---|
| **Task** | Classify one skin lesion into one of 11 diagnoses from its dermoscopic photo |
| **Model** | ViT-B/16 (ImageNet pre-trained), last 2 encoder blocks + new 11-class head fine-tuned |
| **Input** | One dermoscopic image, any size (resized to 224 × 224) |
| **Output** | A probability for each of the 11 classes |
| **Official score** | **0.360** macro-F1 on the 479 blind MILK10k test lesions (ISIC Challenge) |
| **Local estimate** | 0.428 on my own held-out test split (same metric, 10 classes present) |
| **Weights** | `m5_final/model_m5.pt` (~330 MB, not in git) |
| **Code** | `m5_final/pipeline.py`, `predict.py`, `make_submission.py`, `m5_train.ipynb` |

## Intended use

- **For**: learning and research. A reproducible baseline for the MILK10k benchmark, and an example of
  honest evaluation (lesion-level splits, one test split, error analysis, fairness check).
- **Not for**: diagnosing patients, triage, or any decision about a real person's skin. It has not been
  validated clinically, it misses about half of melanomas (see below), and it was built in a university course in a few months.
  Anyone worried about a skin lesion should see a doctor.

## The 11 classes

Column order of `training_gt.csv`, which is also the output and submission order:

| Code | Diagnosis | Benign / malignant |
|---|---|---|
| AKIEC | actinic keratosis / intraepidermal carcinoma | indeterminate or malignant |
| BCC | basal cell carcinoma | malignant |
| BEN_OTH | other benign proliferations | benign |
| BKL | benign keratinocytic lesion (e.g. seborrhoeic keratosis) | benign |
| DF | dermatofibroma | benign |
| INF | inflammatory and infectious conditions | benign |
| MAL_OTH | other malignant proliferations | malignant |
| MEL | melanoma | malignant |
| NV | melanocytic nevus (mole) | benign |
| SCCKA | squamous cell carcinoma / keratoacanthoma | malignant |
| VASC | vascular lesion | benign |

## Data

- **MILK10k** training set (ISIC Archive): 5,240 lesions, each with one dermoscopic and one clinical close-up photo,
  diagnosis confirmed mostly by histopathology. Licence **CC-BY-NC** (non-commercial use only).
- Only the **dermoscopic** photo is used. No clinical photos, no metadata (age, sex, site).
- **Lesion-level split** (Milestone 1, seed 0), so the two photos of one lesion never end up in different splits:

| Split | Lesions | Used for |
|---|---|---|
| train | 3,667 | training |
| validation | 786 | choosing the epoch, the architecture and the thresholds |
| test | 787 | scored once at the end of each milestone |

- **Very imbalanced**: in training, BCC has 1,744 lesions and MAL_OTH has 7.
  DF, VASC, INF and BEN_OTH have 27-37 each. My test split has **no MAL_OTH** lesions.

## Training

- Backbone: torchvision `vit_b_16`, `IMAGENET1K_V1` weights. ImageNet pre-training is the only external data.
- Trainable: encoder blocks 10 and 11, the final layer norm and a new `Linear(768, 11)` head (14.2 M of 85.8 M weights).
- Loss: cross-entropy with class weights `n / (11 × count)`, so rare classes count more.
- Augmentation: random horizontal and vertical flips.
- Adam, learning rate 1e-4, batch 64, 6 epochs; the epoch with the best validation macro-F1 is kept (epoch 4).
- Same architecture and recipe as the Milestone 4 winner (chosen there on 3-class validation macro-F1: ViT 0.663 vs ResNet-18 0.648).

**Validation transform** (identical in training notebook, `predict()` and `make_submission.py`, all from `pipeline.py`):
RGB → resize to 224 × 224 (PIL default) → scale to [0, 1] → ImageNet mean/std normalisation.
Checked in the notebook: `predict()` gives the same probabilities as the notebook (difference ~1e-7).

**Thresholds for the leaderboard.** The official metric binarises each class at 0.5. One threshold per class was tuned
on **validation only** (grid 0.05–0.95; classes with fewer than 5 validation lesions keep 0.5), and the submitted
probabilities are rescaled so each threshold sits at 0.5. Thresholds: BCC, BKL, MEL 0.20 · INF, NV 0.25 ·
AKIEC, BEN_OTH, MAL_OTH, VASC 0.50 · DF 0.55 · SCCKA 0.60. `predict()` returns the plain softmax (no thresholds).

## Results

Macro-F1; my test split was opened once.

| | Validation | My test split | Official blind test |
|---|---|---|---|
| argmax (highest class wins) | 0.373 | 0.405 | — |
| official rule (each class ≥ 0.5) | 0.371 | 0.387 | — |
| official rule + tuned thresholds | 0.435* | **0.428** | **0.360** |
| always "BCC" (majority) | — | 0.068 | — |

\* optimistic: the thresholds were tuned on validation.

Per class on my test split (argmax):

| Class | Lesions | Precision | Recall | F1 |
|---|---|---|---|---|
| BCC | 401 | 0.86 | 0.76 | **0.81** |
| NV | 102 | 0.72 | 0.77 | 0.74 |
| VASC | 8 | 0.50 | 0.75 | 0.60 |
| SCCKA | 73 | 0.51 | 0.66 | 0.58 |
| MEL | 63 | 0.48 | 0.48 | **0.48** |
| AKIEC | 44 | 0.26 | 0.45 | 0.33 |
| DF | 7 | 0.21 | 0.57 | 0.31 |
| BKL | 77 | 0.44 | 0.10 | 0.17 |
| BEN_OTH | 6 | 0.03 | 0.17 | 0.05 |
| INF | 6 | 0.00 | 0.00 | 0.00 |

**Why the official score (0.360) is lower than my estimate (0.428).**
1. My test split has no MAL_OTH, so 0.428 averages 10 classes; the leaderboard averages 11. The model never gives
   MAL_OTH ≥ 0.5 (7 training examples), so if the blind set has any, that class scores 0: 0.428 × 10/11 ≈ 0.39.
2. The rare classes have only a handful of lesions in 479, so one or two mistakes move their F1 a lot; their
   thresholds were tuned on 8–12 validation lesions.
3. The blind lesions are new patients: some drop from a held-out estimate is normal.

## Limitations and risks

- **Misses melanoma.** Recall 0.48: about half of the test melanomas are missed, most of them called moles (NV).
  For a skin-cancer tool this is the most dangerous error.
- **Rare classes barely work** (INF, BEN_OTH, MAL_OTH): too few training examples.
- **Leans on colour.** Error analysis (Milestone 4) suggests "brown = benign, pink = malignant": melanomas that look
  like moles are missed, pale seborrhoeic keratoses are called cancer.
- **AKIEC is partly decided under the microscope**, not in the photo, so it can't be fully separated from BCC/SCCKA.
- **Fairness** (checked in Milestone 4 on the 3-class version of the same architecture, not re-run for 11 classes):
  malignant recall 0.61 for patients under 50 vs 0.87-0.88 for older patients, driven by missed melanomas;
  0.80 for women vs 0.87 for men. **Skin tone** (`skin_tone_class`, from `supplements/training_input.csv`): malignant
  recall is lowest for tone 2 (0.43), but most of those misses are melanomas, the same weakness again. Tones 0-1 have
  0 and 11 test lesions and 61 % of the data is tone 3, so do not assume it works on skin tones that are rare here.
- **One photo type**: trained on dermoscopic images only; phone photos or clinical close-ups are outside what it saw.
- One training run, one seed: differences of approximately 0.01 between runs are noise.

## How to reproduce

From the repo root, with the environment from `requirements.txt` and the MILK10k training data in `data/milk10k/`:

1. **Splits** (Milestone 1): `python m1_data/run_milestone1.py` → `outputs/splits/`.
2. **Train** (Colab, T4 GPU, ~10 min): open `m5_final/m5_train.ipynb`, Runtime -> Run all.
   It writes `model_m5.pt`, `results.json`, `test_per_class.csv`, `thresholds.json`, `local_scores.csv`
   and the confusion matrix. Put them in `m5_final/`.
3. **Thresholds** (if not done in Colab): `python m5_final/tune_thresholds.py`.
4. **Predict one image**: `python m5_final/predict.py path/to/dermoscopic.jpg`
   (or `from m5_final.predict import predict; predict("path.jpg")`).
5. **Leaderboard file**: download `MILK10k_Test_Input.zip` and `MILK10k_Test_Metadata.csv` from
   challenge.isic-archive.com/data, unzip into `data/milk10k_test/`, then
   `python m5_final/make_submission.py --metadata data/milk10k_test/MILK10k_Test_Metadata.csv --images data/milk10k_test/MILK10k_Test_Input`
   → `m5_final/submission.csv` (479 rows, `lesion_id` + 11 classes).
   Note: the course slides say the first column is `lesion`; the ISIC system rejects that and requires `lesion_id`.

## Citation

- MILK study team. *MILK10k Benchmark.* ISIC Archive, 2025. doi:10.34970/262082
- Tschandl P, et al. *MILK10k: A Hierarchical Multimodal Imaging-Learning Toolkit for Diagnosing Pigmented and
  Nonpigmented Skin Cancer and its Simulators.* J Invest Dermatol. 2026;146(2):357-364. doi:10.1016/j.jid.2025.06.1594
- Dosovitskiy A, et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021 (ViT).

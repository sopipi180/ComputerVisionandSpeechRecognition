import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo folder

# Single place where the data folder is defined (no hard-coded absolute paths)
DATA_DIR = Path(os.environ.get("MILK10K_DIR", ROOT / "data" / "milk10k"))
IMG_DIR = DATA_DIR / "images"
GT_PATH = DATA_DIR / "supplements" / "training_gt.csv"

TARGET = "diagnosis_1"  # Benign / Indeterminate / Malignant
SEED = 0                # one seed for the whole project

# The 3 diagnosis_1 classes, in ONE order used by every session.
# Alphabetical, so the order never depends on who typed the list.
# The spelling must match metadata.csv exactly.
CLASSES_3 = ["Benign", "Indeterminate", "Malignant"]

# The 11 classes, in the exact column order of training_gt.csv.
# The leaderboard file must use this order, so it is written once, here.
CLASSES_11 = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
              "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
IMAGE_SIZE = 224        # model input size

# Generated files (never mixed with the source code)
OUTPUT_DIR = ROOT / "outputs"
SPLIT_DIR = OUTPUT_DIR / "splits"
FIG_DIR = OUTPUT_DIR / "figures"


def image_path(isic_id):
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = IMG_DIR / f"{isic_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for isic_id={isic_id!r} in {IMG_DIR}")
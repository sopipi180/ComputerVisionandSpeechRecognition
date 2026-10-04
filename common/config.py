import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo folder

# Single place where the data folder is defined (no hard-coded absolute paths)
DATA_DIR = Path(os.environ.get("MILK10K_DIR", ROOT / "Session1" / "milk10k"))
IMG_DIR = DATA_DIR / "images"
GT_PATH = DATA_DIR / "supplements" / "training_gt.csv"

TARGET = "diagnosis_1"  # Benign / Malignant / Indeterminate
SEED = 0                # one seed for the whole project
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
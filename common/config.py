from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]  # the repo folder
 
# Session 1 downloaded the dataset into Session1/milk10k
DATA_DIR = ROOT / "Session1" / "milk10k"
IMG_DIR = DATA_DIR / "images"
GT_PATH = DATA_DIR / "supplements" / "training_gt.csv"
 
TARGET = "diagnosis_1"  # Benign / Malignant / Indeterminate
 
 
def image_path(isic_id):
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = IMG_DIR / f"{isic_id}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for isic_id={isic_id!r} in {IMG_DIR}")
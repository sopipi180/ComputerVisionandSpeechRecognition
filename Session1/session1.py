import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_URL = "https://isic-archive.s3.amazonaws.com/dois/10.34970-648456/milk10k.zip"
ZIP_PATH = BASE_DIR / "milk10k.zip"
DATA_DIR = BASE_DIR / "milk10k"
IMG_DIR = DATA_DIR / "images"

FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)    


def _save_and_show(filename):
    """Save the current figure to figures/<filename>, then display it."""
    plt.savefig(FIG_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()


SUBCLASS_COLS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
                 "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]


def _show_progress(block_num, block_size, total_size):
    """Called repeatedly by urlretrieve so the download isn't silent."""
    downloaded = block_num * block_size
    mb_done = downloaded / (1024 * 1024)
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb_total = total_size / (1024 * 1024)
        print(f"\rDownloading: {pct:5.1f}%  ({mb_done:,.0f} MB / {mb_total:,.0f} MB)", end="")
    else:
        print(f"\rDownloading: {mb_done:,.0f} MB", end="")


def ensure_dataset():

    if (DATA_DIR / "metadata.csv").exists():
        return

    if ZIP_PATH.exists() and not zipfile.is_zipfile(ZIP_PATH):
        print("Existing milk10k.zip is corrupt/incomplete - deleting and re-downloading.")
        ZIP_PATH.unlink()

    if not ZIP_PATH.exists():
        print("Downloading dataset (~345 MB, 10,480 images)...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH, reporthook=_show_progress)
        print("\nDownload complete.")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print("Extraction complete.")


def load_data():
    ensure_dataset()
    df = pd.read_csv(DATA_DIR / "metadata.csv")
    gt = pd.read_csv(DATA_DIR / "supplements" / "training_gt.csv")
    return df, gt


def show_samples(df, img_dir, n=6, diagnosis=None, seed=0):
    subset = df if diagnosis is None else df[df["diagnosis_1"] == diagnosis]

    sample = subset.sample(n=n, random_state=seed)

    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3))

    for ax, (_, row) in zip(axes, sample.iterrows()):
        img = Image.open(img_dir / f'{row["isic_id"]}.jpg')
        ax.imshow(img)
        ax.set_title(row["diagnosis_1"], fontsize=8)
        ax.axis("off")          # hide pixel-coordinate ticks - they're noise here

    plt.tight_layout()
    plt.show()


def plot_class_counts(df):
    """EDA 1 - number of samples per class (diagnosis_1)."""
    class_counts = df["diagnosis_1"].value_counts()
    print(class_counts)

    plt.figure(figsize=(6, 4))
    class_counts.plot(kind="bar", color="steelblue")
    plt.title("Number of samples per class (diagnosis_1)")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.tight_layout()
    _save_and_show("eda1_class_counts.png")


def plot_subclass_counts(gt):
    """EDA 2 - number of samples per subclass (the 11 diagnoses)."""
    subclass_counts = gt[SUBCLASS_COLS].sum().sort_values(ascending=False)
    print(subclass_counts)

    plt.figure(figsize=(9, 4))
    subclass_counts.plot(kind="bar", color="darkorange")
    plt.title("Number of samples per subclass (11 diagnoses)")
    plt.xlabel("Subclass")
    plt.ylabel("Number of lesions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    _save_and_show("eda2_subclass_counts.png")


def plot_age_range(df):
    """EDA 3 - age range per class."""
    age_stats = df.groupby("diagnosis_1")["age_approx"].agg(["min", "max", "mean", "count"])
    print(age_stats)

    plt.figure(figsize=(6, 4))
    df.boxplot(column="age_approx", by="diagnosis_1")
    plt.title("Age range per class")
    plt.suptitle("")         
    plt.xlabel("Class")
    plt.ylabel("Age (approx.)")
    plt.tight_layout()
    _save_and_show("eda3_age_range.png")


def plot_sex_distribution(df):
    """EDA 4 - sex distribution per class."""
    sex_by_class = df.groupby(["diagnosis_1", "sex"]).size().unstack(fill_value=0)
    print(sex_by_class)

    sex_by_class.plot(kind="bar", figsize=(6, 4))
    plt.title("Sex distribution per class")
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.xticks(rotation=0)
    plt.legend(title="Sex")
    plt.tight_layout()
    _save_and_show("eda4_sex_distribution.png")

if __name__ == "__main__":
    df, gt = load_data()

    n_images = len(list(IMG_DIR.glob("*.jpg")))
    print("Number of images:", n_images)
    print(df.shape, gt.shape)
    print(df["lesion_id"].nunique(), "unique lesions")

    # Teacher's structure checks (slide 76)
    print(df[["age_approx", "sex", "anatom_site_general"]].isna().mean())
    print("Every lesion has exactly 2 images:", (df.groupby("lesion_id").size() == 2).all())

    plot_class_counts(df)
    plot_subclass_counts(gt)
    plot_age_range(df)
    plot_sex_distribution(df)
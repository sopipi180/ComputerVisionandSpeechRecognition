"""
make_submission.py -> submission.csv for the official MILK10k leaderboard.

One row per lesion: "lesion" + the 11 class probabilities, header in training_gt.csv order.
The dermoscopic photo of each lesion is used, as in training. If m5_final/thresholds.json
exists (from tune_thresholds.py), each class is rescaled so its tuned threshold sits at 0.5.

From the repo root:
    # the official blind test set (479 lesions), unzipped from the ISIC Challenge site:
    python m5_final/make_submission.py --metadata MILK10k_Test_Metadata.csv --images MILK10k_Test_Input

    # my own test split (only to check that the pipeline works):
    python m5_final/make_submission.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from common import config
from m5_final.pipeline import CLASSES, load_image, prep, load_model, rescale

HERE = Path(__file__).resolve().parent


def find_image(folder, lesion_id, isic_id):
    """The blind test set has one subfolder per lesion; the training data is one flat folder."""
    for sub in (folder / lesion_id, folder):
        for ext in (".jpg", ".jpeg", ".png"):
            if (sub / f"{isic_id}{ext}").exists():
                return sub / f"{isic_id}{ext}"
    raise FileNotFoundError(f"No image for {isic_id} (lesion {lesion_id}) in {folder}")


def one_photo_per_lesion(meta):
    """The dermoscopic photo of each lesion (the model never saw clinical photos)."""
    if "image_type" in meta.columns:
        meta = meta[meta["image_type"].str.startswith("derm")]
    return meta.drop_duplicates("lesion_id").reset_index(drop=True)


def check(sub, n_lesions):
    """The four checks from the brief, plus one row per lesion."""
    assert list(sub.columns) == ["lesion"] + CLASSES
    assert len(sub) == n_lesions and sub.lesion.is_unique
    assert sub[CLASSES].isna().sum().sum() == 0
    assert ((sub[CLASSES] >= 0) & (sub[CLASSES] <= 1)).all().all()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=config.SPLIT_DIR / "test.csv")
    parser.add_argument("--images", type=Path, default=config.IMG_DIR)
    parser.add_argument("--out", type=Path, default=HERE / "submission.csv")
    parser.add_argument("--no-thresholds", action="store_true", help="plain softmax, ignore thresholds.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    meta = pd.read_csv(args.metadata)
    lesions = one_photo_per_lesion(meta)
    print(f"{len(lesions)} lesions ({len(meta)} rows) from {args.metadata.name}, on {device}")

    probs = []
    with torch.no_grad():
        for i in range(0, len(lesions), 32):
            batch = lesions.iloc[i:i + 32]
            xb = torch.stack([load_image(find_image(args.images, l, s)) for l, s in zip(batch["lesion_id"], batch["isic_id"])])
            probs.append(model(prep(xb.to(device))).softmax(1).cpu())
            print(f"  {min(i + 32, len(lesions))}/{len(lesions)}", end="\r")
    probs = torch.cat(probs)

    thresholds_file = HERE / "thresholds.json"
    if thresholds_file.exists() and not args.no_thresholds:
        thresholds = [json.loads(thresholds_file.read_text())[c] for c in CLASSES]
        probs = rescale(probs, thresholds)
        print("\napplied thresholds.json:", dict(zip(CLASSES, thresholds)))
    else:
        print("\nno thresholds: plain softmax probabilities")

    sub = pd.DataFrame(probs.numpy(), columns=CLASSES)
    sub.insert(0, "lesion", lesions["lesion_id"])
    check(sub, meta["lesion_id"].nunique())
    if "Test_Metadata" in args.metadata.name:
        assert len(sub) == 479, f"the official test set has 479 lesions, got {len(sub)}"
    sub.to_csv(args.out, index=False, float_format="%.6f")
    print(f"checks passed - wrote {args.out} ({len(sub)} rows x {len(sub.columns)} columns)")


if __name__ == "__main__":
    main()

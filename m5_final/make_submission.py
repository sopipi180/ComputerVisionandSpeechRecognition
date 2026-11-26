"""
make_submission.py -> submission.csv: one row per lesion, lesion_id + the probability
of each of the 11 classes, in training_gt.csv column order.

From the repo root:
    python m5_final/make_submission.py                                      # my own test split (checks the pipeline)
    python m5_final/make_submission.py --metadata TEST_METADATA.csv --images TEST_IMAGE_FOLDER

The metadata CSV needs the columns isic_id and lesion_id (and image_type, if the
lesion has more than one photo: the dermoscopic one is used, as in training).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from common import config
from m5_final.pipeline import CLASSES, load_image, prep, load_model


def find_image(folder, isic_id):
    for ext in (".jpg", ".jpeg", ".png"):
        if (folder / f"{isic_id}{ext}").exists():
            return folder / f"{isic_id}{ext}"
    raise FileNotFoundError(f"No image for {isic_id} in {folder}")


def one_photo_per_lesion(meta):
    """The dermoscopic photo of each lesion (the model never saw clinical photos)."""
    if "image_type" in meta.columns:
        meta = meta.sort_values("image_type", key=lambda s: ~s.str.startswith("derm"))   # dermoscopic first
    return meta.drop_duplicates("lesion_id").sort_values("lesion_id").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metadata", type=Path, default=config.SPLIT_DIR / "test.csv")
    parser.add_argument("--images", type=Path, default=config.IMG_DIR)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "submission.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    lesions = one_photo_per_lesion(pd.read_csv(args.metadata))
    print(f"{len(lesions)} lesions from {args.metadata.name}, on {device}")

    probs = []
    with torch.no_grad():
        for i in range(0, len(lesions), args.batch_size):
            ids = lesions["isic_id"].iloc[i:i + args.batch_size]
            xb = torch.stack([load_image(find_image(args.images, isic_id)) for isic_id in ids])
            probs.append(model(prep(xb.to(device))).softmax(1).cpu())
            print(f"  {min(i + args.batch_size, len(lesions))}/{len(lesions)}", end="\r")

    sub = pd.DataFrame(torch.cat(probs).numpy(), columns=CLASSES)
    sub.insert(0, "lesion_id", lesions["lesion_id"])
    sub.to_csv(args.out, index=False, float_format="%.6f")
    print(f"\nwrote {args.out} ({len(sub)} rows, columns: lesion_id + {len(CLASSES)} classes)")


if __name__ == "__main__":
    main()

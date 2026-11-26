"""
tune_thresholds.py - the leaderboard's metric on MY split, and one threshold per class.

The official score binarises every probability at 0.5, computes F1 per class and
averages the 11 F1s. With softmax outputs a rare class seldom reaches 0.5, so the
plain argmax macro-F1 overestimates the official score. This script:

  1. scores validation and test the OFFICIAL way (threshold 0.5), for an honest local estimate;
  2. tunes one threshold per class on VALIDATION only (never on test);
  3. scores test again with those thresholds (test is only reported, never used to choose);
  4. saves m5_final/thresholds.json, which make_submission.py then applies.

From the repo root (needs model_m5.pt and the MILK10k images; ~5 min on a laptop CPU):
    python m5_final/tune_thresholds.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from common import config
from m5_final.pipeline import CLASSES, load_image, prep, load_model, rescale

OUT = Path(__file__).resolve().parent
CACHE = OUT / "_cache"
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)    # thresholds tried
MIN_VAL_LESIONS = 5                                 # fewer positives in validation -> too noisy to tune, keep 0.5


def split_probs(model, name, device):
    """Probabilities for every dermoscopic lesion of one split (cached: computing them is the slow part)."""
    f = CACHE / f"probs_{name}.pt"
    df = pd.read_csv(config.SPLIT_DIR / f"{name}.csv")
    df = df[df["image_type"].str.startswith("derm")].reset_index(drop=True)
    y = np.array([CLASSES.index(c) for c in df["dx11"]])
    if not f.exists():
        CACHE.mkdir(exist_ok=True)
        probs = []
        with torch.no_grad():
            for i in range(0, len(df), 32):
                xb = torch.stack([load_image(config.image_path(j)) for j in df["isic_id"][i:i + 32]])
                probs.append(model(prep(xb.to(device))).softmax(1).cpu())
                print(f"  {name}: {min(i + 32, len(df))}/{len(df)}", end="\r")
        torch.save(torch.cat(probs), f)
        print()
    return torch.load(f).numpy(), y


def official_f1(probs, y):
    """Leaderboard metric: binarise each class at 0.5, F1 per class, plain mean over the classes present in y."""
    present = np.unique(y)
    f1s = [f1_score(y == k, probs[:, k] >= 0.5, zero_division=0) for k in present]
    return float(np.mean(f1s)), dict(zip([CLASSES[k] for k in present], np.round(f1s, 3)))


def argmax_f1(probs, y):
    return float(f1_score(y, probs.argmax(1), labels=np.unique(y), average="macro"))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    if not ((CACHE / "probs_val.pt").exists() and (CACHE / "probs_test.pt").exists()):
        model = load_model(device=device)
    p_val, y_val = split_probs(model, "val", device)
    p_test, y_test = split_probs(model, "test", device)

    # tune on VALIDATION: for each class, the threshold with the best F1 for that class
    thresholds = []
    for k, c in enumerate(CLASSES):
        if (y_val == k).sum() < MIN_VAL_LESIONS:
            thresholds.append(0.5)
            continue
        scores = [f1_score(y_val == k, p_val[:, k] >= t, zero_division=0) for t in GRID]
        thresholds.append(float(GRID[int(np.argmax(scores))]))

    rows = []
    for name, p, y in [("validation", p_val, y_val), ("test", p_test, y_test)]:
        rows.append({"split": name,
                     "argmax macro-F1": round(argmax_f1(p, y), 3),
                     "official-style (0.5)": round(official_f1(p, y)[0], 3),
                     "official-style, tuned thresholds": round(official_f1(rescale(p, thresholds).numpy(), y)[0], 3)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print("\nthresholds:", dict(zip(CLASSES, thresholds)))
    print("test F1 per class, tuned:", official_f1(rescale(p_test, thresholds).numpy(), y_test)[1])

    (OUT / "thresholds.json").write_text(json.dumps(dict(zip(CLASSES, thresholds)), indent=2))
    table.to_csv(OUT / "local_scores.csv", index=False)
    print(f"\nwrote {OUT / 'thresholds.json'} and {OUT / 'local_scores.csv'}")
    print("Note: validation was used to tune, so its tuned score is optimistic; quote the TEST one.")


if __name__ == "__main__":
    main()

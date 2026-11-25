import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "common" / "config.py").is_file())  # the repo folder
SESSION5 = ROOT / "m2_baseline" / "Session5"   # Session 6 reuses its features
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SESSION5))

from common import config          # noqa: E402

CLASSES = config.CLASSES_3
TARGET = config.TARGET
PIXEL_SIZE = 32
PIXEL_CACHE = HERE / f"pixels_{PIXEL_SIZE}.npz"
FEATURE_CACHE = SESSION5 / "features.csv"

# the 33 the slides use: colour statistics + colour histograms + texture
SLIDES_33 = (["mean_R", "std_R", "mean_G", "std_G", "mean_B", "std_B"]
             + [f"hist_{c}{b}" for c in "RGB" for b in range(8)]
             + ["edge_density", "roughness", "laplacian_var"])


def load_splits(view="dermoscopic"):
    """The Session 3 split files, filtered to one kind of photo."""
    out = {}
    for name in ("train", "val", "test"):
        path = config.SPLIT_DIR / f"{name}.csv"
        if not path.exists():
            sys.exit(f"missing {path} - run scripts/run_milestone1.py first")
        df = pd.read_csv(path)
        if view:
            df = df[df["image_type"].str.startswith(view[:4])]
        out[name] = df.reset_index(drop=True)
    return out


# 33 features
def build_features(ids):
    """
    Compute the 33 features straight from the images.

    Only used when Session5/features.csv is missing - on Colab, for instance,
    because the cache is not in git. Takes a couple of minutes.
    """
    sys.path.insert(0, str(SESSION5))
    import features as FT

    ids = list(ids)
    print(f"computing the 33 features for {len(ids)} images ...")
    rows = []
    for n, isic in enumerate(ids, 1):
        row = FT.extract(config.image_path(isic))
        row["isic_id"], row["variant"] = isic, "full frame"
        rows.append(row)
        if n % 1000 == 0:
            print(f"  {n}/{len(ids)}")
    out = pd.DataFrame(rows)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(FEATURE_CACHE, index=False)
    print(f"wrote {FEATURE_CACHE}")
    return out


def load_features(columns=None, ids=None):
    """
    The Session 5 feature table, indexed by image id.

    Only the full-frame rows: Session 5 measured that cropping did not help.
    If the Session 5 cache is missing, the features are computed from the
    images instead, so this also works on a fresh Colab session.
    """
    if FEATURE_CACHE.exists():
        df = pd.read_csv(FEATURE_CACHE)
    elif ids is not None:
        df = build_features(ids)
    else:
        sys.exit(f"missing {FEATURE_CACHE} - run python m2_baseline/Session5/session5.py first, "
                 f"or call load_features(ids=...) to compute them here")
    if "variant" in df.columns:
        df = df[df["variant"] == "full frame"].drop(columns="variant")
    df = df.drop_duplicates("isic_id").set_index("isic_id")
    missing = [c for c in (columns or SLIDES_33) if c not in df.columns]
    if missing:
        sys.exit(f"features.csv has no column(s): {missing}")
    return df[columns or SLIDES_33]


# raw pixels
def load_pixels(ids, size=PIXEL_SIZE, refresh=False):
    """
    Every photo shrunk to size x size x 3 and flattened, scaled to 0-1.

    3,072 numbers per photo with no processing at all: this is what question 3
    feeds the network instead of the 33 designed features. Cached, because
    decoding thousands of JPEGs takes a few minutes.
    """
    ids = list(ids)
    cache = {}
    if PIXEL_CACHE.exists() and not refresh:
        z = np.load(PIXEL_CACHE, allow_pickle=False)
        cache = {"ids": list(z["ids"]), "X": z["X"]}

    known = {i: n for n, i in enumerate(cache.get("ids", []))}
    todo = [i for i in ids if i not in known]
    if todo:
        print(f"decoding {len(todo)} images at {size}x{size} ...")
        rows = []
        for n, isic in enumerate(todo, 1):
            with Image.open(config.image_path(isic)) as im:
                im.draft("RGB", (size * 4, size * 4))     # fast rough decode
                arr = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR))
            rows.append(arr.reshape(-1).astype(np.float32) / 255.0)
            if n % 1000 == 0:
                print(f"  {n}/{len(todo)}")
        new_X = np.stack(rows)
        X = np.vstack([cache["X"], new_X]) if cache else new_X
        all_ids = cache.get("ids", []) + todo
        np.savez_compressed(PIXEL_CACHE, ids=np.array(all_ids), X=X)
        cache = {"ids": all_ids, "X": X}
        print(f"wrote {PIXEL_CACHE}")

    index = {i: n for n, i in enumerate(cache["ids"])}
    return cache["X"][[index[i] for i in ids]]


# labels
def feature_matrix(split_df, feats):
    """The feature rows for one split, in the split's order."""
    missing = set(split_df["isic_id"]) - set(feats.index)
    if missing:
        sys.exit(f"{len(missing)} images in this split are not in features.csv "
                 f"(e.g. {sorted(missing)[0]}).\nRun the full extraction: "
                 f"python m2_baseline/Session5/session5.py")
    return feats.loc[split_df["isic_id"]].to_numpy(dtype="float32")


def encode(labels, classes=CLASSES):
    """Class names -> 0, 1, 2 (the order in CLASSES)."""
    lookup = {c: i for i, c in enumerate(classes)}
    return np.array([lookup[v] for v in labels], dtype=np.int64)


def class_weights(y, n_classes=len(CLASSES)):
    """
    The weights from the slides: n / (n_classes * n_c).

    A class with few examples gets a big weight, so the network cannot improve
    its loss by ignoring it. Same idea as class_weight="balanced" in Session 5.
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    return len(y) / (n_classes * np.maximum(counts, 1))
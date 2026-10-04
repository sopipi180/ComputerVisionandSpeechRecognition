import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

from common import config


def split_lesions(lesions, val_size=0.15, test_size=0.15, seed=config.SEED,
                  label_col="dx", n_splits=20):
    """
    Return (train_ids, val_ids, test_ids): three lists of lesion_id.

    StratifiedGroupKFold keeps the class mix similar in each fold (stratified)
    while never splitting a group (here one lesion = one group, so both of its
    photos always land together).

    It cuts the data into n_splits equal folds (20 folds = 5 % each), then takes
    whole folds for val and test. Asking for a 15 % test set therefore means
    "3 of the 20 folds", which lands much closer to 15 % than doing two separate
    cuts would.
    """
    ids = lesions["lesion_id"].to_numpy()
    y = lesions[label_col].to_numpy()

    k_test = max(1, round(test_size * n_splits))
    k_val = max(1, round(val_size * n_splits))
    if k_test + k_val >= n_splits:
        raise ValueError("val_size + test_size is too large for this n_splits")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        # a class rarer than n_splits cannot appear in every fold - expected here
        warnings.simplefilter("ignore", UserWarning)
        folds = [test_idx for _, test_idx in sgkf.split(lesions, y, groups=ids)]

    test_idx = np.concatenate(folds[:k_test])
    val_idx = np.concatenate(folds[k_test:k_test + k_val])
    train_idx = np.concatenate(folds[k_test + k_val:])

    return (sorted(ids[train_idx]), sorted(ids[val_idx]), sorted(ids[test_idx]))


def check_splits(train_ids, val_ids, test_ids, lesions=None,
                 val_size=0.15, test_size=0.15, tol_pp=1.0):
    """
    Property test (A3.2): no overlap, sizes within tol_pp percentage points.
    Returns a dict of the measured sizes; raises AssertionError if something is off.
    """
    s_tr, s_va, s_te = set(train_ids), set(val_ids), set(test_ids)
    assert not (s_tr & s_va), "train and val share lesions"
    assert not (s_tr & s_te), "train and test share lesions"
    assert not (s_va & s_te), "val and test share lesions"

    total = len(s_tr) + len(s_va) + len(s_te)
    if lesions is not None:
        assert total == len(lesions), "some lesions are in no split"

    got = {"train": len(s_tr) / total, "val": len(s_va) / total, "test": len(s_te) / total}
    want = {"train": 1 - val_size - test_size, "val": val_size, "test": test_size}
    for k in want:
        off = abs(got[k] - want[k]) * 100
        assert off <= tol_pp, f"{k} is {got[k]:.1%}, wanted {want[k]:.0%} (off by {off:.1f} pp)"
    return got


def assign_images(df, lesion_ids):
    """All images whose lesion is in lesion_ids (2 per lesion)."""
    return df[df["lesion_id"].isin(set(lesion_ids))].reset_index(drop=True)


def split_summary(lesions, splits, label_col="dx"):
    """Class proportions (%) per split + the largest deviation from the global mix."""
    global_mix = lesions[label_col].value_counts(normalize=True) * 100
    table = {"all": global_mix}
    for name, ids in splits.items():
        sub = lesions[lesions["lesion_id"].isin(set(ids))]
        table[name] = sub[label_col].value_counts(normalize=True) * 100
    out = pd.DataFrame(table).fillna(0).round(2)
    out["max_dev_pp"] = (out[list(splits)].sub(out["all"], axis=0)).abs().max(axis=1).round(2)
    return out.sort_values("all", ascending=False)


def save_splits(df, splits, out_dir=None):
    """Write one CSV of images per split. Returns {name: path}."""
    out_dir = out_dir or config.SPLIT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, ids in splits.items():
        path = out_dir / f"{name}.csv"
        assign_images(df, ids).to_csv(path, index=False)
        paths[name] = path
    return paths


# A3.3 
def compare_fold_strategies(lesions, df, label_col="dx", image_label_col="dx11",
                            n_splits=5, seed=config.SEED, rare_class="MAL_OTH"):
    """
    Compare GroupKFold / StratifiedKFold (on images) / StratifiedGroupKFold:
    leaked lesions, spread of class proportions across validation folds, and how
    often the rarest class is missing from a validation fold.
    """
    rows = []
    y_les, groups = lesions[label_col].to_numpy(), lesions["lesion_id"].to_numpy()

    strategies = {
        "GroupKFold (groups only)": (GroupKFold(n_splits=n_splits), "lesion"),
        "StratifiedKFold (images, no groups)": (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), "image"),
        "StratifiedGroupKFold (both)": (
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), "lesion"),
    }

    for name, (splitter, level) in strategies.items():
        leaked, spreads, missing_rare = 0, {}, 0
        if level == "image":
            y_img = df[image_label_col].to_numpy()
            folds = splitter.split(df, y_img)
            for tr, va in folds:
                tr_les = set(df["lesion_id"].iloc[tr])
                va_les = set(df["lesion_id"].iloc[va])
                leaked += len(tr_les & va_les)
                mix = df[image_label_col].iloc[va].value_counts(normalize=True) * 100
                for c, v in mix.items():
                    spreads.setdefault(c, []).append(v)
                missing_rare += int(rare_class not in set(df[image_label_col].iloc[va]))
        else:
            folds = splitter.split(lesions, y_les, groups=groups)
            for tr, va in folds:
                leaked += len(set(groups[tr]) & set(groups[va]))
                mix = lesions[label_col].iloc[va].value_counts(normalize=True) * 100
                for c, v in mix.items():
                    spreads.setdefault(c, []).append(v)
                missing_rare += int(rare_class not in set(lesions[label_col].iloc[va]))

        worst = max((max(v) - min(v)) for v in spreads.values())
        rows.append({"strategy": name, "leaked_lesions": leaked,
                     "max_class_spread_pp": round(worst, 2),
                     f"folds_without_{rare_class}": missing_rare})
    return pd.DataFrame(rows)
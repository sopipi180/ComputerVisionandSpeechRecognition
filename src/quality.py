"""Integrity and data-quality checks (A1.1, B1, B4)."""

import pandas as pd
from PIL import Image, UnidentifiedImageError

from common import config
from common.milk10k import load_metadata

# B4: columns that must never be used as model inputs (they leak the label)
LEAKY_COLUMNS = {
    "diagnosis_2": "finer level of the diagnosis",
    "diagnosis_3": "finer level of the diagnosis",
    "diagnosis_4": "finer level of the diagnosis",
    "dx11": "the 11-class label itself",
    "melanocytic": "derived from the diagnosis",
    "diagnosis_confirm_type": "how the diagnosis was confirmed (biopsy = already suspicious)",
    "concomitant_biopsy": "whether a biopsy was taken (already suspicious)",
}

# B4: what to do with every column that has missing values
MISSING_POLICY = {
    "age_approx": "impute with the TRAIN median (a real value we would also have at prediction time)",
    "anatom_site_general": "fill with 'unknown' - being unrecorded is itself informative",
    "anatom_site_special": "drop - 98 % missing",
    "diagnosis_3": "not used as input (leaky)",
    "diagnosis_4": "not used as input (leaky)",
    "melanocytic": "not used as input (leaky)",
}


#  A1.1 
def check_label_files(df, gt):
    """Cross-check metadata.csv and training_gt.csv. Returns a dict of findings."""
    out = {}
    class_cols = [c for c in gt.columns if c != "lesion_id"]

    # (a) same lesions in both files
    meta_ids, gt_ids = set(df["lesion_id"]), set(gt["lesion_id"])
    out["only_in_metadata"] = sorted(meta_ids - gt_ids)
    out["only_in_gt"] = sorted(gt_ids - meta_ids)

    # (b) exactly one positive class per lesion
    positives = gt[class_cols].sum(axis=1)
    out["bad_one_hot"] = gt.loc[positives != 1, "lesion_id"].tolist()

    # (c) does each 11-class map to exactly one diagnosis_1?
    lesion_dx1 = df.drop_duplicates("lesion_id")[["lesion_id", "diagnosis_1"]]
    merged = gt.assign(dx=gt[class_cols].idxmax(axis=1)).merge(lesion_dx1, on="lesion_id")
    mapping = (merged.groupby(["dx", "diagnosis_1"]).size()
                     .rename("lesions").reset_index())
    out["class_mapping"] = mapping
    out["ambiguous_classes"] = (mapping.groupby("dx").size().loc[lambda s: s > 1].index.tolist())

    # (d) diagnosis hierarchy: each child belongs to exactly one parent
    for child, parent in [("diagnosis_3", "diagnosis_2"), ("diagnosis_2", "diagnosis_1")]:
        if child in df.columns and parent in df.columns:
            n_parents = df.dropna(subset=[child]).groupby(child)[parent].nunique()
            out[f"{child}_with_many_{parent}"] = n_parents[n_parents > 1].to_dict()

    # (e) both images of a lesion agree on the lesion-level fields
    fields = [c for c in ["age_approx", "sex", "anatom_site_general", "diagnosis_1",
                          "diagnosis_2", "diagnosis_3"] if c in df.columns]
    disagree = df.groupby("lesion_id")[fields].nunique(dropna=False).gt(1).any(axis=1)
    out["lesions_with_disagreeing_fields"] = df["lesion_id"].loc[
        df["lesion_id"].isin(disagree[disagree].index)].unique().tolist()
    return out


# B1 
def integrity_check(df=None, verify_images=True, limit=None):
    """
    Every row resolves to a file, every file opens, and the image sizes.
    Returns (summary dict, size table). Slow part is opening each image.
    """
    df = load_metadata() if df is None else df
    ids = df["isic_id"] if limit is None else df["isic_id"].head(limit)

    missing, unreadable, sizes = [], [], []
    for iid in ids:
        try:
            path = config.image_path(iid)
        except FileNotFoundError:
            missing.append(iid)
            continue
        if not verify_images:
            continue
        try:
            with Image.open(path) as im:
                im.verify()                      # catches truncated / corrupt files
            with Image.open(path) as im:
                sizes.append({"isic_id": iid, "width": im.width, "height": im.height})
        except (UnidentifiedImageError, OSError):
            unreadable.append(iid)

    size_table = pd.DataFrame(sizes)
    summary = {"rows_checked": len(ids), "missing_files": len(missing),
               "unreadable_files": len(unreadable),
               "missing_ids": missing[:20], "unreadable_ids": unreadable[:20]}
    if len(size_table):
        summary.update({
            "width_min": int(size_table["width"].min()),
            "width_median": float(size_table["width"].median()),
            "width_max": int(size_table["width"].max()),
            "height_min": int(size_table["height"].min()),
            "height_median": float(size_table["height"].median()),
            "height_max": int(size_table["height"].max()),
            "share_at_least_224": float((size_table[["width", "height"]].min(axis=1) >= 224).mean()),
        })
    return summary, size_table


def save_size_summary(size_table, path=None):
    path = path or config.OUTPUT_DIR / "image_size_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    (size_table.groupby(["width", "height"]).size().rename("images")
     .reset_index().sort_values("images", ascending=False)).to_csv(path, index=False)
    return path


# B4 
def shortcut_crosstabs(df, target=config.TARGET):
    """Row-% cross-tabs of the two acquisition fields against the label."""
    out = {}
    for col in ["image_manipulation", "image_type"]:
        if col in df.columns:
            out[col] = (pd.crosstab(df[col].fillna("none"), df[target], normalize="index")
                        * 100).round(1)
    return out


def data_quality_report(df, gt, path=None):
    """Write the Markdown report B4 asks for. Returns the path."""
    path = path or config.OUTPUT_DIR / "data_quality_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = (df.isna().mean() * 100).round(1)
    missing = missing[missing > 0].sort_values(ascending=False)
    per_lesion = df.groupby("lesion_id").size()
    types_ok = df.groupby("lesion_id")["image_type"].nunique().eq(2)
    checks = check_label_files(df, gt)

    lines = ["# Data-quality report (Milestone 1)", "",
             f"- images: {len(df):,}   lesions: {df['lesion_id'].nunique():,}", ""]

    lines += ["## Missing values and what we do with them", "",
              "| column | % missing | decision |", "|---|---|---|"]
    for col, pct in missing.items():
        lines.append(f"| {col} | {pct} | {MISSING_POLICY.get(col, 'leave as is')} |")

    lines += ["", "## Label consistency", "",
              f"- lesions with exactly 2 images: {int((per_lesion == 2).sum()):,} "
              f"of {len(per_lesion):,}",
              f"- lesions with one image of each type: {int(types_ok.sum()):,} "
              f"of {len(types_ok):,}",
              f"- lesions in metadata but not in training_gt: {len(checks['only_in_metadata'])}",
              f"- lesions with more than one positive class: {len(checks['bad_one_hot'])}",
              f"- 11-classes that map to several diagnosis_1 values: "
              f"{checks['ambiguous_classes'] or 'none'}", ""]

    lines += ["## Shortcut check (acquisition fields vs the label)", ""]
    for col, tab in shortcut_crosstabs(df).items():
        lines += [f"### {col} (row %)", "", tab.to_markdown(), ""]
    lines += ["Conclusion: image_type is identical across classes by construction "
              "(every lesion has one of each), so it carries no label information, "
              "but it does change image statistics. image_manipulation does differ "
              "between classes, so it is treated as an acquisition shortcut and is "
              "not used as a model input.", ""]

    lines += ["## Columns NOT used as model inputs (label leakage)", ""]
    for col, why in LEAKY_COLUMNS.items():
        if col in df.columns:
            lines.append(f"- `{col}` - {why}")
    lines.append("- `image_manipulation`, `attribution`, `copyright_license` - "
                 "acquisition / provenance, not the lesion")

    path.write_text("\n".join(lines))
    return path
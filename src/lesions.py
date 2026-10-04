import pandas as pd
from common import config
from common.milk10k import load_metadata

DERM = "dermoscopic"
CLINICAL = "clinical: close-up"


def build_lesion_table(df=None):
    """
    Per-image metadata -> one row per lesion with:
        lesion_id, derm_id, clinical_id, diagnosis_1, dx (11-class), age, sex, site
    Built with pivot/groupby only (no loops over rows).
    """
    if df is None:
        df = load_metadata()

    # the isic_id of each view, side by side: pivot image_type into columns
    ids = (df.pivot_table(index="lesion_id", columns="image_type",
                          values="isic_id", aggfunc="first")
             .rename(columns={DERM: "derm_id", CLINICAL: "clinical_id"}))

    # fields that are identical for both images of a lesion -> take the first
    per_lesion = (df.groupby("lesion_id")
                    .agg(diagnosis_1=("diagnosis_1", "first"),
                         dx=("dx11", "first"),
                         age=("age_approx", "first"),
                         sex=("sex", "first"),
                         site=("anatom_site_general", "first")))

    lesions = ids.join(per_lesion).reset_index()
    return lesions[["lesion_id", "derm_id", "clinical_id", "diagnosis_1", "dx",
                    "age", "sex", "site"]]


def check_lesion_table(lesions, expected_rows=None):
    """Assertions asked for in A1.3. expected_rows=5240 on the full dataset."""
    assert lesions["lesion_id"].is_unique, "duplicate lesion_id"
    assert lesions["derm_id"].notna().all(), "lesion without a dermoscopic image"
    assert lesions["clinical_id"].notna().all(), "lesion without a clinical image"
    if expected_rows is not None:
        assert len(lesions) == expected_rows, f"{len(lesions)} rows, expected {expected_rows}"
    return True


def save_lesion_table(lesions, path=None):
    path = path or config.OUTPUT_DIR / "lesions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lesions.to_csv(path, index=False)
    return path
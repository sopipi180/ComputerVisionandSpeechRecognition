import pandas as pd
 
from . import config
 
 
def load_metadata():
    """metadata.csv (one row per image) + a 'dx11' column with the 11-class label."""
    df = pd.read_csv(config.DATA_DIR / "metadata.csv")
    gt = pd.read_csv(config.GT_PATH)
    gt["dx11"] = gt.drop(columns="lesion_id").idxmax(axis=1)  # one-hot -> label
    return df.merge(gt[["lesion_id", "dx11"]], on="lesion_id", how="left")
 
 
def available_subset(df):
    """Keep only rows whose image exists on disk."""
    def exists(isic_id):
        try:
            config.image_path(isic_id)
            return True
        except FileNotFoundError:
            return False
    return df[df["isic_id"].apply(exists)].reset_index(drop=True)
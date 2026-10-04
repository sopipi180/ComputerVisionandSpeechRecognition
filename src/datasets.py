import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common import config
from src import labels as label_utils
from src.transforms import eval_transform, train_transform


def _load_rgb(isic_id, draft_size=None):
    """
    Open one image. Fails loudly (B1): a missing file is a bug, not a row to skip.

    draft_size asks the JPEG decoder for a smaller image directly (PIL's draft
    mode). Decoding at 1/2 or 1/4 scale is several times faster than decoding
    the full 600x450 and resizing afterwards, which matters when a cell reads
    hundreds of images. Use it only where the result is downscaled anyway.
    """
    with Image.open(config.image_path(isic_id)) as im:
        if draft_size:
            im.draft("RGB", (draft_size, draft_size))
        return im.convert("RGB")


class MilkImageDataset(Dataset):
    """
    One item = (image_tensor, label, isic_id).
    table: a split CSV (per image) with isic_id and the label column.
    label_map: {class name: int} from label_map.json.
    """

    def __init__(self, table, label_map, label_col=config.TARGET, transform=None,
                 draft_size=None):
        self.table = table.reset_index(drop=True)
        self.label_map = label_map
        self.label_col = label_col
        self.transform = transform or eval_transform()
        self.draft_size = draft_size

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        image = self.transform(_load_rgb(row["isic_id"], self.draft_size))
        return image, self.label_map[row[self.label_col]], row["isic_id"]


class LesionDataset(Dataset):
    """
    A3.6 - one item = one LESION.
    Returns {"derm": tensor, "clinical": tensor, "label": int, "lesion_id": str}
    (only the requested views). The transform is applied independently to each
    view, so the two photos get different random augmentations.
    """

    def __init__(self, lesion_table, label_map, label_col="dx", transform=None,
                 views=("derm", "clinical"), draft_size=None):
        self.table = lesion_table.reset_index(drop=True)
        self.label_map = label_map
        self.label_col = label_col
        self.transform = transform or eval_transform()
        self.views = tuple(views)
        self.draft_size = draft_size
        for v in self.views:
            if f"{v}_id" not in self.table.columns:
                raise ValueError(f"lesion table has no column {v}_id")

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        row = self.table.iloc[i]
        item = {"label": self.label_map[row[self.label_col]], "lesion_id": row["lesion_id"]}
        for view in self.views:
            isic_id = row[f"{view}_id"]
            if pd.isna(isic_id):
                raise FileNotFoundError(f"lesion {row['lesion_id']} has no {view} image")
            item[view] = self.transform(_load_rgb(isic_id, self.draft_size))  # per view
        return item


def build_dataloaders(split_tables, label_map, label_col=config.TARGET, batch_size=32,
                      num_workers=0, seed=config.SEED, use_sampler=False, size=config.IMAGE_SIZE):
    """
    B8 - train (shuffled, augmented) + val/test (ordered, deterministic).
    use_sampler=True draws rare classes more often (WeightedRandomSampler)
    instead of relying only on class weights in the loss.
    """
    set_seeds(seed)
    loaders = {}
    for name, table in split_tables.items():
        is_train = name == "train"
        ds = MilkImageDataset(table, label_map, label_col,
                              transform=train_transform(size) if is_train else eval_transform(size))
        if is_train and use_sampler:
            w = label_utils.sample_weights(table[label_col].tolist())
            sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                            num_samples=len(w), replacement=True)
            loaders[name] = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                                       num_workers=num_workers)
        else:
            loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=is_train,
                                       num_workers=num_workers)
    return loaders


def set_seeds(seed=config.SEED):
    """Same numbers every run (torch, numpy, random)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def aggregate_predictions(image_probs, image_table, id_col="isic_id",
                          lesion_col="lesion_id"):
    """
    A3.6c - per-image class probabilities -> ONE prediction per lesion, by
    averaging the two views.
    image_probs: DataFrame indexed by isic_id, one column per class.
    Returns a DataFrame indexed by lesion_id with the mean probabilities and
    the predicted class.
    """
    mapping = image_table.set_index(id_col)[lesion_col]
    probs = image_probs.copy()
    probs[lesion_col] = probs.index.map(mapping)
    out = probs.groupby(lesion_col).mean()
    out["prediction"] = out.idxmax(axis=1)
    return out
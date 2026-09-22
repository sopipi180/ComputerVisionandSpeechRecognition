from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import config


def to_gray(arr):
    """Luminosity formula from class: Gray = 0.299R + 0.587G + 0.114B (same as cv2)."""
    return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)


def preprocess_image(src, size=(224, 224), color="rgb", norm="minmax"):
    """
    Task 9. src = file path or numpy array.
    size:  (height, width), or None to keep the original size
    color: "rgb" or "gray"
    norm:  "minmax" -> divide by 255, values in [0, 1]  (default: keeps real
                       brightness differences between images)
           "zscore" -> per image: mean 0, std 1 (removes lighting differences)
           None     -> leave as 0-255 integers
    Returns (array, info).
    """
    img = Image.open(src) if isinstance(src, (str, Path)) else Image.fromarray(np.asarray(src))
    img = img.convert("RGB")
    original = (img.height, img.width)

    if size:
        img = img.resize((size[1], size[0]))  # PIL wants (width, height)
    arr = np.array(img)
    if color == "gray":
        arr = to_gray(arr)

    if norm == "minmax":
        arr = arr.astype(np.float32) / 255
    elif norm == "zscore":
        arr = arr.astype(np.float32)
        arr = (arr - arr.mean(axis=(0, 1))) / (arr.std(axis=(0, 1)) + 1e-6)

    info = {"original_size": original, "shape": arr.shape, "color": color, "norm": norm,
            "min": round(float(arr.min()), 3), "max": round(float(arr.max()), 3)}
    return arr, info


def process_batch(items, **options):
    """
    Task 10. items = list of paths, or a metadata DataFrame (uses its isic_id).
    Bad files are skipped, not fatal.
    Returns (images, ids, skipped) - images is one stacked array if all shapes match.
    """
    if isinstance(items, pd.DataFrame):
        ids = list(items["isic_id"])
        sources = ids
    else:
        sources = list(items)
        ids = [Path(p).stem for p in sources]

    images, kept, skipped = [], [], []
    for id_, src in zip(ids, sources):
        try:
            path = config.image_path(src) if isinstance(items, pd.DataFrame) else src
            arr, _ = preprocess_image(path, **options)
            images.append(arr)
            kept.append(id_)
        except Exception as e:
            skipped.append((id_, type(e).__name__))

    if images and len({a.shape for a in images}) == 1:
        images = np.stack(images)
    return images, kept, skipped
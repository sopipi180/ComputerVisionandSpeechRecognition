import json
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from common import config

# B3 decision - primary target: keep the three classes.
# Indeterminate is small (2.3 %) but clinically it means "the doctor could not
# decide", which is exactly the case a tool must not silently call benign.
DIAGNOSIS_1_CLASSES = ["Benign", "Indeterminate", "Malignant"]

# B3 decision - 11-class stretch: keep all 11 classes and handle the rare ones
# with class weights / a sampler instead of merging them. Merging DF, INF, VASC,
# BEN_OTH and MAL_OTH into one "other" bucket would mix benign and malignant
# lesions in a single label, which is worse than a class the model rarely sees.
RARE_CLASSES = ["DF", "INF", "VASC", "BEN_OTH", "MAL_OTH"]


def build_label_map(labels, name):
    """{class name: integer} - sorted so the mapping is stable across runs."""
    return {"target": name, "classes": {c: i for i, c in enumerate(sorted(set(labels)))}}


def save_label_map(label_maps, path=None):
    path = path or config.OUTPUT_DIR / "label_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label_maps, indent=2))
    return path


def load_label_map(path=None):
    path = path or config.OUTPUT_DIR / "label_map.json"
    return json.loads(path.read_text())


def class_weights(train_labels, classes=None):
    """
    Weight for each class in the loss: rare class -> bigger weight.
    Computed on the TRAIN split only (never peek at val/test).
    """
    classes = np.array(classes if classes is not None else sorted(set(train_labels)))
    w = compute_class_weight("balanced", classes=classes, y=np.asarray(train_labels))
    return {str(c): round(float(v), 4) for c, v in zip(classes, w)}


def sample_weights(train_labels):
    """
    One weight per ROW, for WeightedRandomSampler: 1 / (how many rows share
    this label), so every class is drawn about equally often.
    """
    counts = pd.Series(train_labels).value_counts()
    return np.array([1.0 / counts[l] for l in train_labels], dtype=float)


def save_class_weights(weights, path=None):
    path = path or config.OUTPUT_DIR / "class_weights.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(weights, indent=2))
    return path


def imbalance_ratio(labels):
    """Biggest class count / smallest class count."""
    counts = pd.Series(labels).value_counts()
    return float(counts.max() / counts.min())
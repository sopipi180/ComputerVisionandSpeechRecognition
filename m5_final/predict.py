"""
predict(image_path) -> the probability of each of the 11 MILK10k classes.

From the repo root:
    python m5_final/predict.py path/to/image.jpg

From Python:
    from m5_final.predict import predict
    predict("path/to/image.jpg")      # {"AKIEC": 0.01, "BCC": 0.83, ...}

Give it the DERMOSCOPIC photo of a lesion: that is what the model was trained on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # so "m5_final" and "common" import

import torch
from m5_final.pipeline import CLASSES, load_image, prep, load_model

_model = None                                                      # loaded once, on the first call


def predict(image_path, model=None):
    """Load the model (once), apply exactly the validation transform, return {class: probability}."""
    global _model
    if model is None:
        if _model is None:
            _model = load_model()
        model = _model
    device = next(model.parameters()).device
    x = load_image(image_path).unsqueeze(0)                        # a batch of one image
    with torch.no_grad():
        p = model(prep(x.to(device))).softmax(1)[0].cpu()
    return {c: float(v) for c, v in zip(CLASSES, p)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python m5_final/predict.py path/to/image.jpg")
    probs = predict(sys.argv[1])
    for c, p in sorted(probs.items(), key=lambda kv: -kv[1]):     # most likely first
        print(f"{c:8s} {p:.3f}")

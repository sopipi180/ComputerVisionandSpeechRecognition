"""
The ONE place where the final model and its validation transform are defined.

The training notebook (m5_train.ipynb), predict.py and make_submission.py all
import from here, so the transform used at prediction time is exactly the one
used on the validation set - they cannot drift apart.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights

ROOT = Path(__file__).resolve().parents[1]          # the repo folder
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from common import config

CLASSES = config.CLASSES_11                         # training_gt.csv column order = submission column order
MODEL_PATH = Path(__file__).resolve().parent / "model_m5.pt"

# ImageNet statistics: the backbone learned on ImageNet (Session 7)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_image(path):
    """Validation transform, step 1: any image file -> uint8 tensor (3, 224, 224). Same as Milestone 4."""
    img = Image.open(path).convert("RGB").resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
    return torch.from_numpy(np.array(img)).permute(2, 0, 1)        # (H, W, 3) -> (3, H, W)


def prep(xb):
    """Validation transform, step 2: uint8 batch -> [0, 1] -> ImageNet-normalised, on xb's device."""
    return (xb.float() / 255 - MEAN.to(xb.device)) / STD.to(xb.device)


def build_model(pretrained=True):
    """ViT-B/16 (the Milestone 4 winner) with a new last layer: 11 outputs instead of 3."""
    net = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    net.heads = nn.Linear(768, len(CLASSES))
    return net


def load_model(path=MODEL_PATH, device="cpu"):
    """The trained 11-class model, ready to predict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The weights (~330 MB) are not in git: "
            f"download model_m5.pt (see MODEL_CARD.md) and put it in {path.parent}/")
    model = build_model(pretrained=False)          # no ImageNet download needed: our weights replace them
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval().to(device)

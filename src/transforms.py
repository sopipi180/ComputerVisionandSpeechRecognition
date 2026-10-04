import torchvision.transforms as T

from common import config

# ImageNet statistics: the usual choice when fine-tuning a pretrained backbone.
# Milestone 2 can swap these for statistics computed on the TRAIN split only.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

AUGMENTATION_TABLE = [
    ("RandomHorizontalFlip", "p=0.5", "a mirrored lesion is the same lesion"),
    ("RandomVerticalFlip", "p=0.5", "skin images have no up/down convention"),
    ("RandomRotation", "degrees=15", "the camera angle is arbitrary"),
    ("RandomResizedCrop", "size=224, scale=(0.85, 1.0)", "scale varies; 0.85 keeps the lesion in frame"),
    ("ColorJitter", "brightness=0.2, contrast=0.2, saturation=0.1, hue=0", "lighting differs between clinics; hue is left untouched because colour is diagnostic"),
    ("Normalize", "ImageNet mean/std", "matches the statistics pretrained backbones expect"),
]


def train_transform(size=config.IMAGE_SIZE):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0),
        T.RandomResizedCrop(size, scale=(0.85, 1.0), antialias=True),
        T.ToTensor(),
        T.Normalize(NORM_MEAN, NORM_STD),
    ])


def eval_transform(size=config.IMAGE_SIZE):
    """Deterministic: same image in -> same tensor out, every time."""
    return T.Compose([
        T.Resize((size, size), antialias=True),
        T.ToTensor(),
        T.Normalize(NORM_MEAN, NORM_STD),
    ])


def is_deterministic(transform, image):
    """Proof for B7: applying it twice gives exactly the same tensor."""
    import torch
    return bool(torch.equal(transform(image), transform(image)))
import math
import numpy as np
import matplotlib.pyplot as plt
from common.config import TARGET, FIG_DIR

SHOW = False  

CLASS_COLORS = {"Benign": "#2a78d6", "Malignant": "#eb6834", "Indeterminate": "#1baf7a"}


def color_of(label):
    return CLASS_COLORS.get(str(label), "#8a8984")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name, fig_dir=None):
    """Save a figure, then show or close it."""
    fig_dir = fig_dir or FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / name, dpi=150, bbox_inches="tight")
    plt.show() if SHOW else plt.close(fig)


def displayable(img):
    """Any image (uint8, 0-1 floats, normalised tensor, CHW or HWC) -> 0-1 HWC."""
    a = np.asarray(img.detach().cpu() if hasattr(img, "detach") else img)
    if a.ndim == 3 and a.shape[0] in (1, 3):      # torch CHW -> HWC
        a = a.transpose(1, 2, 0)
    if a.shape[-1] == 1:
        a = a[..., 0]
    if a.dtype == np.uint8:
        return a / 255
    a = a.astype(np.float32)
    if a.max() > 1 or a.min() < 0:                # e.g. after Normalize
        a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return a


def show_grid(images, titles=None, ncols=6, title=None, filename="grid.png", fig_dir=None):
    """Grid of images with a title above each."""
    images = list(images)
    nrows = math.ceil(len(images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows), squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < len(images):
            img = displayable(images[i])
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            if titles is not None:
                ax.set_title(str(titles[i]), fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save(fig, filename, fig_dir)


def plot_class_balance(df, label_col=TARGET, filename="class_balance.png", logy=False,
                       title=None, fig_dir=None):
    """Images (or lesions) per class as a bar chart. Returns the counts."""
    counts = df[label_col].value_counts()
    colors = [CLASS_COLORS.get(c, "#2a78d6") for c in counts.index]

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(counts) + 2), 4))
    ax.bar(counts.index.astype(str), counts.values, color=colors, width=0.6)
    if logy:
        ax.set_yscale("log")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("count" + (" (log scale)" if logy else ""))
    ax.set_title(title or f"Class balance ({label_col}): biggest / smallest = "
                          f"{counts.max() / counts.min():.0f}x", fontsize=10)
    if len(counts) > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    style(ax)
    save(fig, filename, fig_dir)
    return counts


def plot_batch_check(images, raw=None, filename="batch_check.png", fig_dir=None):
    """Pixel-value histogram of a batch (did normalisation work?) + raw vs processed."""
    x = np.asarray(images.detach().cpu() if hasattr(images, "detach") else images,
                   dtype=np.float32)
    k = min(4, len(x)) if raw is not None else 0
    fig, axes = plt.subplots(2, k + 1, figsize=(2.4 * k + 5, 4.8),
                             gridspec_kw={"width_ratios": [1] * k + [2.5]}, squeeze=False)
    for i in range(k):
        axes[0, i].imshow(displayable(raw[i]))
        axes[1, i].imshow(displayable(x[i]))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    if k:
        axes[0, 0].set_title("raw", fontsize=8, loc="left")
        axes[1, 0].set_title("processed", fontsize=8, loc="left")

    hist_ax = fig.add_subplot(1, k + 1, k + 1)
    axes[0, k].axis("off")
    axes[1, k].axis("off")
    hist_ax.hist(x.ravel(), bins=60, color="#52514e")
    hist_ax.set_title(f"pixel values in batch  min={x.min():.2f}  max={x.max():.2f}  "
                      f"mean={x.mean():.2f}", fontsize=9)
    style(hist_ax)
    save(fig, filename, fig_dir)
    return {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
            "std": float(x.std()), "shape": tuple(x.shape)}
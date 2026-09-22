import math
import numpy as np
import matplotlib.pyplot as plt
from common.config import TARGET
from plotting import CLASS_COLORS, style, save


def displayable(img):
    """Turn any image into 0-1 values so it can be shown."""
    a = np.asarray(img)
    if a.dtype == np.uint8:  # raw 0-255 image
        return a / 255
    a = a.astype(np.float32)
    if a.max() > 1 or a.min() < 0:  # e.g. z-score

        a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    return a


def show_grid(images, titles=None, ncols=6, title=None, filename="grid.png"):
    """Task 15: grid of images with a title under each."""
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
    save(fig, filename)


def plot_class_balance(df, label_col=TARGET, filename="class_balance.png"):
    """Task 16: images per class as a bar chart."""
    counts = df[label_col].value_counts()
    colors = [CLASS_COLORS.get(c, "#2a78d6") for c in counts.index]

    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(counts) + 2), 4))
    ax.bar(counts.index.astype(str), counts.values, color=colors, width=0.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n{v / counts.sum():.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_ylabel("images")
    ax.set_title(f"Class balance ({label_col}): biggest / smallest = "
                 f"{counts.max() / counts.min():.0f}x", fontsize=10)
    if len(counts) > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    style(ax)
    save(fig, filename)
    return counts


def plot_batch_check(images, raw=None, filename="batch_check.png"):
    """Task 17: pixel-value histogram of a batch (did normalisation work?) + raw vs processed."""
    x = np.asarray(images, dtype=np.float32)
    k = min(4, len(x)) if raw is not None else 0
    fig, axes = plt.subplots(2, k + 1, figsize=(2.4 * k + 5, 4.8),
                             gridspec_kw={"width_ratios": [1] * k + [2.5]}, squeeze=False)

    for i in range(k):
        axes[0, i].imshow(raw[i])
        axes[1, i].imshow(displayable(x[i]))
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    if k:
        axes[0, 0].set_title("raw", fontsize=8, loc="left")
        axes[1, 0].set_title("processed", fontsize=8, loc="left")

    hist_ax = fig.add_subplot(1, k + 1, k + 1)
    axes[0, k].axis("off")
    axes[1, k].axis("off")
    if x.ndim == 4:
        for c, (name, col) in enumerate(zip("RGB", ["#e34948", "#008300", "#2a78d6"])):
            hist_ax.hist(x[..., c].ravel(), bins=60, histtype="step", linewidth=2,
                         color=col, label=name)
        hist_ax.legend(frameon=False)
    else:
        hist_ax.hist(x.ravel(), bins=60, color="#52514e")
    hist_ax.set_title(f"pixel values in batch  min={x.min():.2f}  max={x.max():.2f}  "
                      f"mean={x.mean():.2f}", fontsize=9)
    style(hist_ax)
    save(fig, filename)
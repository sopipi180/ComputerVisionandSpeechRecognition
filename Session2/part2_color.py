import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from common import config
from common.config import TARGET
from plotting import color_of, style, save
from part3_processing import to_gray

STATS = ["mean_R", "mean_G", "mean_B", "std_R", "std_G", "std_B", "mean_gray"]


def sample_per_class(df, n=25, seed=0):
    """Same number per class, half dermoscopic / half clinical (so the camera doesn't bias it)."""
    shuffled = df.sample(frac=1, random_state=seed)
    return shuffled.groupby([TARGET, "image_type"]).head(n // 2).reset_index(drop=True)


def image_colors(path):
    """Histograms (like plt.hist(bins=256), scaled to sum to 1) + mean/std per channel."""
    rgb = np.array(Image.open(path).convert("RGB"))
    gray = to_gray(rgb)
    out = {"hist_gray": np.bincount(gray.ravel(), minlength=256) / gray.size,
           "mean_gray": gray.mean()}
    for i, ch in enumerate("RGB"):
        out[f"hist_{ch}"] = np.bincount(rgb[..., i].ravel(), minlength=256) / gray.size
        out[f"mean_{ch}"] = rgb[..., i].mean()
        out[f"std_{ch}"] = rgb[..., i].std()
    return out


def color_table(sample):
    rows = []
    for _, r in sample.iterrows():
        try:
            rows.append({TARGET: r[TARGET], "image_type": r["image_type"],
                         **image_colors(config.image_path(r["isic_id"]))})
        except Exception as e:
            print("skipped", r["isic_id"], e)
    return pd.DataFrame(rows)


def plot_histograms(t):
    """Task 5: average grayscale and RGB histograms, one line per class."""
    classes = t[TARGET].value_counts().index

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), sharey=True)
    for ax, key in zip(axes, ["hist_gray", "hist_R", "hist_G", "hist_B"]):
        for c in classes:
            ax.plot(np.stack(t.loc[t[TARGET] == c, key]).mean(axis=0),
                    color=color_of(c), linewidth=2, label=c)
        ax.set_title(key.replace("hist_", "").replace("gray", "Grayscale"), fontsize=10)
        ax.set_xlabel("intensity (0-255)")
        style(ax)
    axes[0].set_ylabel("share of pixels")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Average histogram per class", fontsize=11)
    save(fig, "part2_histograms.png")

    # same grayscale plot, split by image type
    types = sorted(t["image_type"].unique())
    fig, axes = plt.subplots(1, len(types), figsize=(6 * len(types), 3.8), sharey=True)
    for ax, typ in zip(np.atleast_1d(axes), types):
        for c in classes:
            sub = t[(t[TARGET] == c) & (t["image_type"] == typ)]
            if len(sub):
                ax.plot(np.stack(sub["hist_gray"]).mean(axis=0), color=color_of(c),
                        linewidth=2, label=c)
        ax.set_title(typ, fontsize=10)
        ax.set_xlabel("grayscale intensity")
        style(ax)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Grayscale histogram per class, split by image type", fontsize=11)
    save(fig, "part2_histograms_by_image_type.png")


def plot_stats(t):
    """Task 6: boxplots of colour statistics per class."""
    classes = list(t[TARGET].value_counts().index)
    fig, axes = plt.subplots(1, len(STATS), figsize=(16, 3.6))
    for ax, s in zip(axes, STATS):
        box = ax.boxplot([t.loc[t[TARGET] == c, s] for c in classes], patch_artist=True)
        for patch, c in zip(box["boxes"], classes):
            patch.set_facecolor(color_of(c))
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(classes) + 1), [c[:5] for c in classes], fontsize=7)
        ax.set_title(s, fontsize=9)
        style(ax)
    fig.suptitle("Colour statistics per image, by class", fontsize=11)
    save(fig, "part2_color_stats.png")


def run(df, log, per_class=25):
    t = color_table(sample_per_class(df, per_class))
    log(f"images analysed: {len(t)}")
    log(pd.crosstab(t[TARGET], t["image_type"]).to_string())

    plot_histograms(t)
    plot_stats(t)

    log("\nMean colour per class:")
    log(t.groupby(TARGET)[STATS].mean().round(1).to_string())
    log("\nMean colour per image type and class (is the image-type gap bigger than the class gap?):")
    log(t.groupby(["image_type", TARGET])[["mean_R", "mean_G", "mean_B", "mean_gray"]]
        .mean().round(1).to_string())
    return t
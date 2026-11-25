import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common" / "config.py").is_file())  # the repo folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import config, milk10k, viz          
import filters as F                              

FIG_DIR = Path(__file__).resolve().parent / "figures"
RESULTS = Path(__file__).resolve().parent / "results.txt"

CLASSES = config.CLASSES_3
SIGMAS = [0, 1, 2, 4, 8, 16]
HALF_SIZE = 300          # all images read at half resolution: same size for everyone
log_lines = []


def log(line=""):
    print(line)
    log_lines.append(str(line))


def load_gray(isic_id, draft=HALF_SIZE):
    with Image.open(config.image_path(isic_id)) as im:
        if draft:
            im.draft("RGB", (draft, draft))
        return F.to_gray(np.asarray(im.convert("RGB")))


# 1. filter bank


def exercise_1(df, n_stats):
    """Apply a bank of filters to images of each class, then measure the classes."""
    log("=" * 70)
    log("1. FILTERS ON DIFFERENT DIAGNOSIS CLASSES")
    log("=" * 70)

    # one dermoscopic example per class (same one every run - seed fixed)
    derm = df[df["image_type"].str.startswith("derm")]
    examples = {}
    for cls in CLASSES:
        rows = derm[derm[config.TARGET] == cls]
        if len(rows):
            examples[cls] = rows.sample(1, random_state=config.SEED)["isic_id"].iloc[0]

    names = list(F.KERNELS)
    fig, axes = plt.subplots(len(examples), len(names) + 1,
                             figsize=(2.0 * (len(names) + 1), 2.3 * len(examples)),
                             squeeze=False)
    for r, (cls, isic) in enumerate(examples.items()):
        gray = load_gray(isic)
        axes[r][0].imshow(gray, cmap="gray")
        axes[r][0].set_ylabel(cls, fontsize=10, color=viz.color_of(cls))
        axes[r][0].set_xticks([])
        axes[r][0].set_yticks([])
        if r == 0:
            axes[r][0].set_title("original (gray)", fontsize=8)
        for c, name in enumerate(names, start=1):
            axes[r][c].imshow(F.stretch(F.apply_kernel(gray, F.KERNELS[name])), cmap="gray")
            axes[r][c].axis("off")
            if r == 0:
                axes[r][c].set_title(name, fontsize=8)
    fig.suptitle("The same filter bank applied to one lesion of each class", y=1.0)
    viz.save(fig, "s4_filter_bank.png", fig_dir=FIG_DIR)
    log(f"example images: {examples}")

    # the actual question: do the classes differ? measure, do not eyeball ---
    stats = {}
    for cls in CLASSES:
        rows = derm[derm[config.TARGET] == cls]
        if rows.empty:
            continue
        ids = rows.sample(min(n_stats, len(rows)), random_state=config.SEED)["isic_id"]
        energy, texture, density = [], [], []
        for isic in ids:
            g = load_gray(isic)
            energy.append(F.edge_energy(g))
            texture.append(F.texture_score(g))
            density.append(F.edge_density(F.canny(g, sigma=2.0)))
        stats[cls] = {"n": len(ids),
                      "edge_energy": np.array(energy),
                      "texture": np.array(texture),
                      "canny_density": np.array(density)}

    log("")
    log(f"{'class':<16}{'n':>5}{'edge energy':>18}{'texture (Lap var)':>22}{'canny edge %':>16}")
    for cls, s in stats.items():
        log(f"{cls:<16}{s['n']:>5}"
            f"{s['edge_energy'].mean():>12.4f} ±{s['edge_energy'].std():.4f}"
            f"{s['texture'].mean():>16.5f} ±{s['texture'].std():.5f}"
            f"{100 * s['canny_density'].mean():>12.2f} ±{100 * s['canny_density'].std():.2f}")

    # bar chart of the three measures
    measures = [("edge_energy", "mean gradient magnitude"),
                ("texture", "variance of the Laplacian"),
                ("canny_density", "fraction of pixels Canny calls an edge")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, label) in zip(axes, measures):
        names_ = list(stats)
        means = [stats[c][key].mean() for c in names_]
        errs = [stats[c][key].std() for c in names_]
        ax.bar(names_, means, yerr=errs, capsize=4,
               color=[viz.color_of(c) for c in names_])
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        viz.style(ax)
    fig.suptitle(f"Edge and texture measures per class "
                 f"(n = {min(n_stats, min(s['n'] for s in stats.values()))} images per class)")
    viz.save(fig, "s4_class_texture.png", fig_dir=FIG_DIR)
    return stats


#  2. Gaussian blur


def exercise_2(df):
    """What sigma does, shown and measured."""
    log("")
    log("=" * 70)
    log("2. GAUSSIAN BLUR - WHAT SIGMA CONTROLS")
    log("=" * 70)
    log("scipy.ndimage.gaussian_filter(input, sigma): sigma is the standard")
    log("deviation of the Gaussian, in PIXELS. It sets how far a pixel is")
    log("smeared into its neighbours. scipy builds a kernel of half-width")
    log("int(truncate*sigma + 0.5), truncate=4.0 by default, so the window is")
    log("2*int(4*sigma + 0.5)+1 pixels wide. Bigger sigma = wider kernel =")
    log("more blur, and the cost grows with sigma.")
    log("")

    isic = df[df["image_type"].str.startswith("derm")].sample(
        1, random_state=config.SEED)["isic_id"].iloc[0]
    gray = load_gray(isic)

    # One fixed pair of thresholds, taken from the unblurred image, so the edge
    # counts below are comparable. (Canny's default thresholds are relative to
    # each image's own strongest gradient, which would hide the effect of blur.)
    peak = F.canny_peak(gray, sigma=1.0)
    fixed = (0.10 * peak, 0.20 * peak)

    fig, axes = plt.subplots(2, len(SIGMAS), figsize=(2.3 * len(SIGMAS), 5.2))
    rows = []
    for c, s in enumerate(SIGMAS):
        blurred = gray if s == 0 else F.gaussian(gray, s)
        width = 1 if s == 0 else F.gaussian_window(s)
        energy = F.edge_energy(blurred, sigma=0)
        edges = F.canny(blurred, sigma=1.0, thresholds=fixed)
        rows.append({"sigma": s, "kernel": width, "edge_energy": energy,
                     "canny_pixels": int(edges.sum()),
                     "detail_kept": float(blurred.std() / gray.std())})
        axes[0][c].imshow(blurred, cmap="gray", vmin=0, vmax=1)
        axes[0][c].set_title(f"sigma = {s}\nkernel {width}x{width} px", fontsize=8)
        axes[0][c].axis("off")
        axes[1][c].imshow(edges, cmap="gray")
        axes[1][c].set_title(f"Canny after: {int(edges.sum())} px", fontsize=8)
        axes[1][c].axis("off")
    fig.suptitle("Top: increasing sigma.  Bottom: the edges that survive it", y=1.0)
    viz.save(fig, "s4_gaussian_sigma.png", fig_dir=FIG_DIR)

    log(f"{'sigma':>7}{'kernel (px)':>14}{'edge energy':>14}{'canny px':>11}{'detail kept':>13}")
    for r in rows:
        log(f"{r['sigma']:>7}{r['kernel']:>14}{r['edge_energy']:>14.4f}"
            f"{r['canny_pixels']:>11}{r['detail_kept']:>13.3f}")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([r["sigma"] for r in rows], [r["edge_energy"] for r in rows], "o-",
            color="#2a78d6", label="edge energy")
    ax.set_xlabel("sigma (pixels)")
    ax.set_ylabel("mean gradient magnitude", color="#2a78d6")
    ax2 = ax.twinx()
    ax2.plot([r["sigma"] for r in rows], [r["canny_pixels"] for r in rows], "s--",
             color="#eb6834")
    ax2.set_ylabel("Canny edge pixels", color="#eb6834")
    ax.set_title("More blur = less detail left to detect")
    viz.style(ax)
    viz.save(fig, "s4_gaussian_effect.png", fig_dir=FIG_DIR)
    return isic, rows


# 3. Canny vs Sobel


def exercise_3(isic):
    """My Canny, step by step, next to Sobel."""
    log("")
    log("=" * 70)
    log("3. CANNY (IMPLEMENTED BY HAND) vs SOBEL")
    log("=" * 70)

    gray = load_gray(isic)
    sigma = 2.0

    # --- the five steps of Canny, drawn one by one ---
    blurred = F.gaussian(gray, sigma)
    mag, angle = F.sobel(gray, sigma=sigma)
    thin = F.non_maximum_suppression(mag, angle)
    peak = thin.max()
    strong, weak = thin >= 0.20 * peak, (thin >= 0.10 * peak) & (thin < 0.20 * peak)
    final = F.canny(gray, sigma=sigma)

    steps = [(gray, "0. grayscale"), (blurred, f"1. blurred, sigma={sigma}"),
             (F.stretch(mag), "2. Sobel gradient magnitude"),
             (F.stretch(thin), "3. after non-maximum suppression"),
             (strong.astype(float) + 0.45 * weak, "4. strong (white) + weak (gray)"),
             (final, "5. after hysteresis = final Canny")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (im, title) in zip(axes.ravel(), steps):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Canny, one step at a time", y=0.98)
    viz.save(fig, "s4_canny_steps.png", fig_dir=FIG_DIR)

    # --- side by side with Sobel ---
    sobel_mag, _ = F.sobel(gray, sigma=sigma)
    sobel_map = F.sobel_edges(gray, sigma=sigma)
    panels = [(gray, "grayscale"), (F.stretch(sobel_mag), "Sobel magnitude (continuous)"),
              (sobel_map, "Sobel, thresholded"), (final, "Canny (my implementation)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (im, title) in zip(axes, panels):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    viz.save(fig, "s4_canny_vs_sobel.png", fig_dir=FIG_DIR)

    # --- zoom, so the thickness difference is actually visible ---
    h, w = gray.shape
    sl = (slice(h // 2 - 60, h // 2 + 60), slice(w // 2 - 60, w // 2 + 60))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (im, title) in zip(axes, [(gray[sl], "zoom: grayscale"),
                                      (sobel_map[sl], "zoom: Sobel"),
                                      (final[sl], "zoom: Canny")]):
        ax.imshow(im, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    viz.save(fig, "s4_canny_vs_sobel_zoom.png", fig_dir=FIG_DIR)

    numbers = {
        "sobel_pixels": int(sobel_map.sum()),
        "canny_pixels": int(final.sum()),
        "sobel_density": F.edge_density(sobel_map),
        "canny_density": F.edge_density(final),
        "nms_dropped": 1 - thin.astype(bool).sum() / max(mag.astype(bool).sum(), 1),
        "weak_pixels": int(weak.sum()),
        "weak_kept": int((weak & final).sum()),
    }
    log(f"Sobel marks {numbers['sobel_pixels']} pixels "
        f"({100 * numbers['sobel_density']:.2f}% of the image)")
    log(f"Canny marks {numbers['canny_pixels']} pixels "
        f"({100 * numbers['canny_density']:.2f}% of the image)")
    log(f"of {numbers['weak_pixels']} weak pixels, hysteresis kept "
        f"{numbers['weak_kept']} (connected to a strong edge)")

    # compare with the reference implementation
    numbers["checked"] = False
    try:
        from skimage.feature import canny as reference_canny
        from scipy import ndimage
        ref = reference_canny(gray, sigma=sigma, low_threshold=0.10,
                              high_threshold=0.20, use_quantiles=False)
        near_ref = ndimage.binary_dilation(ref, np.ones((3, 3)))
        agree = float((final & near_ref).sum() / max(final.sum(), 1))
        numbers.update(checked=True, agreement=agree, ref_pixels=int(ref.sum()))
        log(f"cross-check vs skimage.feature.canny: {100 * agree:.1f}% of my edge "
            f"pixels land within 1 pixel of a reference edge")
    except ImportError:
        log("cross-check skipped (scikit-image not installed - "
            "`pip install scikit-image` to enable it)")
    return numbers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="use 20 images per class for the statistics instead of 60")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = milk10k.available_subset(milk10k.load_metadata())
    log(f"{len(df)} images available, {df['lesion_id'].nunique()} lesions")

    exercise_1(df, n_stats=20 if args.quick else 60)
    example_isic, _ = exercise_2(df)
    exercise_3(example_isic)

    RESULTS.write_text("\n".join(log_lines) + "\n")
    print(f"\nfigures -> {FIG_DIR}\nnumbers -> {RESULTS}")


if __name__ == "__main__":
    main()
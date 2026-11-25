import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "common" / "config.py").is_file())  # the repo folder
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from common import config, viz          
import features as FT                   
import models as M                      

RESULTS = HERE / "results"
FIG_DIR = HERE / "figures"
CACHE = HERE / "features.csv"
LOG = HERE / "results.txt"

CLASSES = config.CLASSES_3
TARGET = config.TARGET

# two ways of looking at the same photo - stage 1 decides which is better
VARIANTS = {"full frame": None, f"centre {int(FT.CROP_FRACTION * 100)}%": FT.CROP_FRACTION}

FEATURE_SETS = {
    "color (6)":        ["color"],
    "histograms (24)":  ["hist"],
    "texture (3)":      ["texture"],
    "glcm (5)":         ["glcm"],
    "color+hist (30)":  ["color", "hist"],
    "slides 33":        ["color", "hist", "texture"],
    "texture+glcm (8)": ["texture", "glcm"],
    "all 40":           ["color", "hist", "texture", "glcm", "shape"],
}

log_lines = []


def log(line=""):
    print(line)
    log_lines.append(str(line))


def banner(title):
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


#  data + features
def load_splits(quick=False):
    """The Session 3 split files -> {'train': df, 'val': df, 'test': df}."""
    out = {}
    for name in ("train", "val", "test"):
        path = config.SPLIT_DIR / f"{name}.csv"
        if not path.exists():
            sys.exit(f"missing {path}\nRun scripts/run_milestone1.py first (Session 3).")
        df = pd.read_csv(path)
        if quick:   # a random sample, NOT the first rows: the file is ordered by class
            keep = (df["lesion_id"].drop_duplicates()
                      .sample(min(400 if name == "train" else 120,
                                  df["lesion_id"].nunique()), random_state=config.SEED))
            df = df[df["lesion_id"].isin(set(keep))]
        out[name] = df.reset_index(drop=True)
    return out


def extract_features(all_ids, refresh=False):
    """
    One row of numbers per image per crop variant, cached on disk.

    Extraction is the slow part (about 20 ms an image), so it runs once and the
    result is reused. Nothing here looks at a label, so doing every split at the
    same time cannot leak anything between them.
    """
    cached = pd.DataFrame()
    if CACHE.exists() and not refresh:
        cached = pd.read_csv(CACHE).drop_duplicates(["isic_id", "variant"])
        log(f"cache: {len(cached)} rows already in {CACHE.name}")

    done = set(map(tuple, cached[["isic_id", "variant"]].to_numpy())) if len(cached) else set()
    todo = [(i, v) for v in VARIANTS for i in all_ids if (i, v) not in done]
    if todo:
        log(f"extracting {len(todo)} rows ({len(all_ids)} images x "
            f"{len(VARIANTS)} crops) ...")
        rows, t0 = [], time.time()
        for n, (isic, variant) in enumerate(todo, 1):
            row = FT.extract(config.image_path(isic), crop=VARIANTS[variant])
            row.update(isic_id=isic, variant=variant)
            rows.append(row)
            if n % 1000 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                log(f"  {n}/{len(todo)}  ({rate:.0f} rows/s, {(len(todo) - n) / rate:.0f}s left)")
        cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
        cached.to_csv(CACHE, index=False)
        log(f"wrote {CACHE}")

    return cached.set_index(["variant", "isic_id"]).sort_index()


def columns_for(set_name, groups):
    return [c for g in FEATURE_SETS[set_name] for c in groups[g]]


def matrix(split_df, feats, columns, variant):
    """The rows of `feats` for this split and this crop, as X, y, lesion ids."""
    sub = feats.loc[variant].loc[split_df["isic_id"]]
    return (sub[columns].to_numpy(), split_df[TARGET].to_numpy(),
            split_df["lesion_id"].to_numpy())


def view(split_df, which):
    """Filter a split to one kind of photo."""
    if which == "both":
        return split_df
    prefix = "derm" if which == "dermoscopic" else "clinical"
    return split_df[split_df["image_type"].str.startswith(prefix)].reset_index(drop=True)

# one fitting
def run(cfg, train_df, eval_df, feats, groups, per_lesion=False):
    """
    Train one configuration on `train_df` and score it on `eval_df`.

    cfg = {model, feature_set, variant, params}. Nothing in here ever sees the
    test split; the caller decides what `eval_df` is.
    """
    estimator, needs_scaling = M.model_zoo()[cfg["model"]]
    pipe = M.build(estimator, needs_scaling)
    if cfg.get("params"):
        pipe.set_params(**cfg["params"])

    columns = columns_for(cfg["feature_set"], groups)
    Xtr, ytr, _ = matrix(train_df, feats, columns, cfg["variant"])
    Xev, yev, lesion_ids = matrix(eval_df, feats, columns, cfg["variant"])

    t0 = time.time()
    pipe.fit(Xtr, ytr)

    if per_lesion:
        pred = M.aggregate_to_lesions(M.scores(pipe, Xev), lesion_ids, pipe.classes_)
        truth = pd.Series(yev, index=lesion_ids).groupby(level=0).first()
        y_true, y_pred = truth.to_numpy(), pred.reindex(truth.index).to_numpy()
    else:
        y_true, y_pred = yev, pipe.predict(Xev)

    row = M.evaluate(y_true, y_pred, CLASSES)
    row.update(model=cfg["model"], feature_set=cfg["feature_set"], variant=cfg["variant"],
               n_features=len(columns), n_train=len(ytr), n_eval=len(y_true),
               seconds=round(time.time() - t0, 1))
    return row, pipe, (y_true, y_pred)


# ========================================= stage 1 - crops x features x models


def stage1(splits, feats, groups):
    banner("STAGE 1 - crop x feature set x model, scored on VALIDATION (dermoscopic)")
    train, val = view(splits["train"], "dermoscopic"), view(splits["val"], "dermoscopic")
    log(f"train {len(train)} images, val {len(val)} images")
    log(f"validation class mix: {dict(pd.Series(val[TARGET]).value_counts())}")

    rows = []
    for variant in VARIANTS:
        for set_name in FEATURE_SETS:
            for model_name in M.model_zoo():
                row, _, _ = run({"model": model_name, "feature_set": set_name,
                                 "variant": variant}, train, val, feats, groups)
                rows.append(row)
        best = max((r for r in rows if r["variant"] == variant), key=lambda r: r["macro_f1"])
        log(f"  {variant:<14} best: {best['model']} on {best['feature_set']} "
            f"-> macro-F1 {best['macro_f1']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "stage1_crop_features_models.csv", index=False)

    # did cropping help? compare the two variants like for like
    pair = table.pivot_table(index=["feature_set", "model"], columns="variant",
                             values="macro_f1")
    full, crop = list(VARIANTS)
    diff = pair[crop] - pair[full]
    log("")
    log(f"cropping vs full frame, over all {len(pair)} feature-set/model combinations:")
    log(f"  crop better in {int((diff > 0).sum())}, worse in {int((diff < 0).sum())}, "
        f"tied in {int((diff == 0).sum())}")
    log(f"  mean macro-F1 {pair[full].mean():.3f} (full) vs {pair[crop].mean():.3f} (crop), "
        f"median difference {diff.median():+.3f}")
    pair.round(3).to_csv(RESULTS / "stage1_crop_comparison.csv")
    best_variant = pair.mean().idxmax()
    log(f"  -> carrying forward the '{best_variant}' images")

    pivot = (table[table["variant"] == best_variant]
             .pivot(index="feature_set", columns="model", values="macro_f1").round(3))
    pivot = pivot.reindex([s for s in FEATURE_SETS if s in pivot.index])
    pivot = pivot[[m for m in M.model_zoo() if m in pivot.columns]]
    pivot.to_csv(RESULTS / "stage1_macro_f1.csv")
    log("")
    log(f"macro-F1 on validation ({best_variant}):")
    log(pivot.to_string())
    plot_stage1(pivot, best_variant)
    plot_crop_comparison(pair, full, crop)
    return table, best_variant

# stage 2 - hyperparameters
def stage2(splits, feats, groups, table, best_variant, top_n=2):
    banner("STAGE 2 - settings: a small grid on VALIDATION")
    train, val = view(splits["train"], "dermoscopic"), view(splits["val"], "dermoscopic")

    sub = table[(table["variant"] == best_variant) & (table["model"] != "majority")]
    sub = sub.sort_values("macro_f1", ascending=False)
    sub = sub.assign(fam=sub["model"].map(M.family)).drop_duplicates("fam").head(top_n)
    grids = M.param_grids()

    rows, best_cfgs = [], []
    for start in sub.to_dict("records"):
        grid = grids.get(M.family(start["model"]), [{}])
        log(f"  tuning {start['model']} on '{start['feature_set']}' "
            f"({len(grid)} settings, default scored {start['macro_f1']:.3f})")
        scored = []
        for params in grid:
            cfg = {"model": start["model"], "feature_set": start["feature_set"],
                   "variant": best_variant, "params": params}
            row, _, _ = run(cfg, train, val, feats, groups)
            row["params"] = str(params)
            scored.append((row, cfg))          # keep the real dict, not its text
            rows.append(row)
        best_row, best_cfg = max(scored, key=lambda pair: pair[0]["macro_f1"])
        gain = best_row["macro_f1"] - start["macro_f1"]
        log(f"    best {best_row['params']} -> macro-F1 {best_row['macro_f1']:.3f} "
            f"({gain:+.3f} vs the default settings)")
        best_cfgs.append(best_cfg)

    pd.DataFrame(rows).to_csv(RESULTS / "stage2_hyperparameters.csv", index=False)
    return best_cfgs

# stage 3 - photos
def stage3(splits, feats, groups, cfgs):
    banner("STAGE 3 - which photos to use (validation)")
    rows = []
    for cfg in cfgs:
        for which, per_lesion in (("dermoscopic", False), ("clinical", False),
                                  ("both", True)):
            row, _, _ = run(cfg, view(splits["train"], which), view(splits["val"], which),
                            feats, groups, per_lesion=per_lesion)
            row.update(views=which, unit="lesion" if per_lesion else "image",
                       params=str(cfg.get("params", {})))
            rows.append(row)
            log(f"  {cfg['model']:<9} {cfg['feature_set']:<18} {which:<12} "
                f"({row['unit']}s)  macro-F1 {row['macro_f1']:.3f}  "
                f"balanced acc {row['balanced_acc']:.3f}")
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "stage3_views.csv", index=False)
    plot_stage3(table)
    return table

# stage 4 - test
def stage4(splits, feats, groups, views_table, cfgs):
    banner("STAGE 4 - the chosen configuration, scored ONCE on TEST")
    win = views_table.sort_values("macro_f1", ascending=False).iloc[0]
    cfg = next(c for c in cfgs if c["model"] == win["model"]
               and c["feature_set"] == win["feature_set"])
    per_lesion = win["unit"] == "lesion"

    log(f"chosen: {win['model']} on '{win['feature_set']}', {win['variant']} images, "
        f"{win['views']} views, judged per {win['unit']}")
    log(f"settings: {cfg.get('params') or 'defaults'}")
    log(f"validation macro-F1 {win['macro_f1']:.3f}")

    # validation has done its job (it chose the configuration), so it is folded
    # back into training rather than thrown away. The test split is still untouched.
    train_full = pd.concat([view(splits["train"], win["views"]),
                            view(splits["val"], win["views"])], ignore_index=True)
    test_df = view(splits["test"], win["views"])

    row, pipe, (y_true, y_pred) = run(cfg, train_full, test_df, feats, groups,
                                      per_lesion=per_lesion)
    maj, _, _ = run({"model": "majority", "feature_set": win["feature_set"],
                     "variant": win["variant"]}, train_full, test_df, feats, groups,
                    per_lesion=per_lesion)

    log(f"refitted on {row['n_train']} images (train+val), tested on "
        f"{row['n_eval']} {win['unit']}s")
    log("")
    log(f"TEST macro-F1      {row['macro_f1']:.3f}   (majority baseline {maj['macro_f1']:.3f})")
    log(f"TEST accuracy      {row['accuracy']:.3f}   (majority baseline {maj['accuracy']:.3f})")
    log(f"TEST balanced acc  {row['balanced_acc']:.3f}")
    log("")
    cm = M.confusion(y_true, y_pred, CLASSES)
    log("confusion matrix (rows = truth, columns = prediction):")
    log(cm.to_string())
    log("")
    per_class = M.per_class_table(y_true, y_pred, CLASSES)
    log("per class:")
    log(per_class.to_string())

    cm.to_csv(RESULTS / "test_confusion_matrix.csv")
    per_class.to_csv(RESULTS / "test_per_class.csv")
    pd.DataFrame([{**row, "params": str(cfg.get("params", {})), "views": win["views"],
                   "unit": win["unit"], "majority_macro_f1": maj["macro_f1"],
                   "majority_accuracy": maj["accuracy"]}]).to_csv(
        RESULTS / "test_result.csv", index=False)
    plot_confusion(cm, win)
    importances(train_full, feats, groups, cfg)


def importances(train_df, feats, groups, cfg):
    """
    Which features carry the signal?

    A random forest is fitted on the chosen feature set only to read its
    feature_importances_ - it describes the data, not the selected model, so
    this runs whichever model actually won.
    """
    forest_cfg = {**cfg, "model": "forest", "params": {}}
    _, pipe, _ = run(forest_cfg, train_df, train_df, feats, groups)
    imp = (pd.Series(pipe[-1].feature_importances_,
                     index=columns_for(cfg["feature_set"], groups))
             .sort_values(ascending=False).round(4))
    imp.to_csv(RESULTS / "feature_importances.csv", header=["importance"])
    log("")
    log("10 most informative features:")
    log(imp.head(10).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    top = imp.head(12)[::-1]
    ax.barh(top.index, top.to_numpy(), color="#2a78d6")
    ax.set_xlabel("importance")
    ax.set_title("Which features carry the signal (random forest)")
    viz.style(ax)
    viz.save(fig, "s5_feature_importances.png", fig_dir=FIG_DIR)

# figures
def plot_stage1(pivot, variant):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = pivot.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=max(0.5, np.nanmax(data)))
    ax.set_xticks(range(pivot.shape[1]), pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(pivot.shape[0]), pivot.index, fontsize=8)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            ax.text(c, r, f"{data[r, c]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if data[r, c] > 0.6 * np.nanmax(data) else "#333")
    ax.set_title(f"Validation macro-F1, {variant} images")
    fig.colorbar(im, ax=ax, shrink=0.8, label="macro-F1")
    viz.save(fig, "s5_stage1_grid.png", fig_dir=FIG_DIR)

def plot_crop_comparison(pair, full, crop):
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(pair[full], pair[crop], s=22, color="#2a78d6", alpha=0.8)
    lo = float(min(pair.min().min(), 0.25))
    hi = float(pair.max().max()) * 1.03
    ax.plot([lo, hi], [lo, hi], "--", color="#8a8984", linewidth=1)
    ax.set_xlabel(f"macro-F1, {full}")
    ax.set_ylabel(f"macro-F1, {crop}")
    ax.set_title("Does cropping to the lesion help?\n(above the line = yes)")
    viz.style(ax)
    viz.save(fig, "s5_crop_comparison.png", fig_dir=FIG_DIR)

def plot_stage3(table):
    table = table.copy()
    table["config"] = table["model"] + "\n" + table["feature_set"]
    pivot = table.pivot_table(index="config", columns="views", values="macro_f1")
    order = [v for v in ("dermoscopic", "clinical", "both") if v in pivot.columns]
    colors = {"dermoscopic": "#2a78d6", "clinical": "#eb6834", "both": "#1baf7a"}

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(pivot))
    width = 0.8 / len(order)
    for k, which in enumerate(order):
        label = "both (one prediction per lesion)" if which == "both" else f"{which} only"
        bars = ax.bar(x + (k - (len(order) - 1) / 2) * width, pivot[which], width,
                      label=label, color=colors[which])
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=8)
    ax.set_ylim(0, max(0.6, float(pivot.to_numpy().max()) * 1.15))
    ax.set_ylabel("macro-F1 (validation)")
    ax.set_title("Which photos to use, and what to predict about")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              frameon=False)
    viz.style(ax)
    viz.save(fig, "s5_stage3_views.png", fig_dir=FIG_DIR)

def plot_confusion(cm, win):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    data = cm.to_numpy()
    ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=9)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            ax.text(c, r, int(data[r, c]), ha="center", va="center", fontsize=10,
                    color="white" if data[r, c] > data.max() * 0.6 else "#333")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"TEST: {win['model']} on {win['feature_set']}")
    viz.save(fig, "s5_test_confusion.png", fig_dir=FIG_DIR)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="a small subset, for testing")
    ap.add_argument("--refresh", action="store_true", help="re-extract every feature")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_splits(quick=args.quick)
    log("splits: " + ", ".join(f"{k} {len(v)} images / {v['lesion_id'].nunique()} lesions"
                               for k, v in splits.items()))
    all_ids = pd.concat([s["isic_id"] for s in splits.values()]).tolist()
    feats = extract_features(all_ids, refresh=args.refresh)
    groups = FT.group_columns([c for c in feats.columns])
    log("feature groups: " + ", ".join(f"{k} {len(v)}" for k, v in groups.items()))

    table, best_variant = stage1(splits, feats, groups)
    cfgs = stage2(splits, feats, groups, table, best_variant)
    views_table = stage3(splits, feats, groups, cfgs)
    stage4(splits, feats, groups, views_table, cfgs)

    LOG.write_text("\n".join(log_lines) + "\n")
    print(f"\ntables  -> {RESULTS}\nfigures -> {FIG_DIR}\nlog     -> {LOG}")


if __name__ == "__main__":
    main()
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import pandas as pd

from common import config, viz
from common.milk10k import load_metadata
from src import labels as label_utils
from src import quality, splits as split_utils
from src.datasets import build_dataloaders, set_seeds
from src.lesions import build_lesion_table, check_lesion_table, save_lesion_table
from src.transforms import AUGMENTATION_TABLE, eval_transform, is_deterministic, train_transform

LOG = []


def log(*parts):
    line = " ".join(str(p) for p in parts)
    print(line)
    LOG.append(line)


def main(quick=False, batch_size=32, num_workers=0):
    set_seeds(config.SEED)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metadata()
    gt = pd.read_csv(config.GT_PATH)
    log(f"metadata: {df.shape}   gt: {gt.shape}   data dir: {config.DATA_DIR}")

    # ---------------------------------------------------------------- B1 ---
    log("\n===== B1  integrity =====")
    summary, size_table = quality.integrity_check(df, verify_images=not quick)
    log(json.dumps({k: v for k, v in summary.items() if not k.endswith("_ids")}, indent=2))
    if len(size_table):
        log("saved:", quality.save_size_summary(size_table))

    # ---------------------------------------------------------------- B2 ---
    log("\n===== B2  label-centred EDA =====")
    lesions = build_lesion_table(df)
    check_lesion_table(lesions)
    log(f"lesion table: {len(lesions):,} rows (expected 5,240 on the full dataset)")
    log("saved:", save_lesion_table(lesions))

    viz.plot_class_balance(lesions, config.TARGET, "b2_balance_diagnosis1.png",
                           title="Lesions per diagnosis_1 class")
    viz.plot_class_balance(lesions, "dx", "b2_balance_11class_log.png", logy=True,
                           title="Lesions per 11-class label (log scale)")
    mapping = quality.check_label_files(df, gt)["class_mapping"]
    mapping.to_csv(config.OUTPUT_DIR / "class_mapping.csv", index=False)
    log("11-class -> diagnosis_1 mapping saved to outputs/class_mapping.csv")

    # one example image per class, 3 x 4 gallery
    gallery = lesions.groupby("dx").head(1)
    from src.datasets import _load_rgb
    viz.show_grid([_load_rgb(i) for i in gallery["derm_id"]],
                  titles=[f"{r.dx}\n({r.diagnosis_1})" for r in gallery.itertuples()],
                  ncols=4, title="One lesion per class (dermoscopic view)",
                  filename="b2_gallery.png")

    # ---------------------------------------------------------------- B3 ---
    log("\n===== B3  label strategy =====")
    label_maps = {
        "diagnosis_1": label_utils.build_label_map(lesions[config.TARGET], config.TARGET),
        "dx11": label_utils.build_label_map(lesions["dx"], "dx"),
    }
    log("saved:", label_utils.save_label_map(label_maps))
    log("diagnosis_1 classes:", label_maps["diagnosis_1"]["classes"])
    log("rare 11-classes kept and handled with weights:", label_utils.RARE_CLASSES)

    # ---------------------------------------------------------------- B4 ---
    log("\n===== B4  data-quality report =====")
    log("saved:", quality.data_quality_report(df, gt))

    # ---------------------------------------------------------------- B5 ---
    log("\n===== B5  splits =====")
    train_ids, val_ids, test_ids = split_utils.split_lesions(lesions, seed=config.SEED)
    sizes = split_utils.check_splits(train_ids, val_ids, test_ids, lesions)
    log("sizes:", {k: f"{v:.1%}" for k, v in sizes.items()})
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

    summary_tbl = split_utils.split_summary(lesions, split_ids)
    log("\nclass mix per split (%):")
    log(summary_tbl.to_string())
    log(f"largest deviation from the global mix: {summary_tbl['max_dev_pp'].max():.2f} pp")

    dx1_tbl = split_utils.split_summary(lesions, split_ids, label_col=config.TARGET)
    log("\ndiagnosis_1 mix per split (%):")
    log(dx1_tbl.to_string())

    for name, ids in split_ids.items():
        images = split_utils.assign_images(df, ids)
        per_lesion = images.groupby("lesion_id").size()
        assert (per_lesion == 2).all(), f"{name}: a lesion does not have exactly 2 images"
    paths = split_utils.save_splits(df, split_ids)
    log("saved:", ", ".join(str(p) for p in paths.values()))

    # ---------------------------------------------------------------- B6 ---
    log("\n===== B6  preprocessing decisions =====")
    if len(size_table):
        share = (size_table[["width", "height"]].min(axis=1) >= config.IMAGE_SIZE).mean()
        log(f"images at least {config.IMAGE_SIZE}px on the short side: {share:.1%}")
    log(f"input resolution: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    log("views: both images of a lesion are used as separate training samples with the "
        "lesion label; evaluation averages the two views per lesion (see aggregate_predictions)")

    # ---------------------------------------------------------------- B7 ---
    log("\n===== B7  augmentation =====")
    example = _load_rgb(lesions["derm_id"].iloc[0])
    log("eval_transform deterministic:", is_deterministic(eval_transform(), example))
    tt = train_transform()
    for cls in list(label_maps["dx11"]["classes"])[:3]:
        row = lesions[lesions["dx"] == cls].iloc[0]
        img = _load_rgb(row["derm_id"])
        viz.show_grid([img] + [tt(img) for _ in range(7)],
                      titles=["original"] + [f"aug {i+1}" for i in range(7)],
                      ncols=4, title=f"Augmentations - {cls}",
                      filename=f"b7_augmentations_{cls}.png")
    pd.DataFrame(AUGMENTATION_TABLE, columns=["augmentation", "parameters", "why it keeps the label"]
                 ).to_csv(config.OUTPUT_DIR / "augmentations.csv", index=False)
    log("saved: outputs/augmentations.csv and 3 augmentation figures")

    # ---------------------------------------------------------------- B8 ---
    log("\n===== B8  dataloaders =====")
    split_tables = {name: pd.read_csv(p) for name, p in paths.items()}
    label_map = label_maps["diagnosis_1"]["classes"]
    train_labels = split_tables["train"][config.TARGET].tolist()

    weights = label_utils.class_weights(train_labels, list(label_map))
    log("class weights (train only):", weights)
    log("imbalance ratio on train:", round(label_utils.imbalance_ratio(train_labels), 1))
    log("saved:", label_utils.save_class_weights(weights))

    loaders = build_dataloaders(split_tables, label_map, batch_size=batch_size,
                                num_workers=num_workers, use_sampler=True)
    images, targets, ids = next(iter(loaders["train"]))
    log(f"batch: {tuple(images.shape)} {images.dtype}  min={images.min():.2f} "
        f"max={images.max():.2f}")

    seen = []
    for i, (_, y, _) in enumerate(loaders["train"]):
        seen += y.tolist()
        if i == 19:
            break
    inv = {v: k for k, v in label_map.items()}
    log("labels over 20 sampled batches:",
        pd.Series([inv[i] for i in seen]).value_counts().to_dict())

    t0 = time.time()
    n = sum(len(y) for _, y, _ in loaders["val"])
    log(f"one pass over val: {n:,} images in {time.time() - t0:.1f}s")

    raw = [_load_rgb(i) for i in ids[:4]]
    stats = viz.plot_batch_check(images, raw=raw, filename="b8_batch_check.png")
    log("batch stats:", {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
    viz.show_grid(images[:16], titles=[inv[int(t)] for t in targets[:16]], ncols=4,
                  title="16 training images after the transforms", filename="b8_train_batch.png")

    (config.OUTPUT_DIR / "milestone1_log.txt").write_text("\n".join(LOG))
    print(f"\nDone. Outputs in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip opening every image in B1")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    main(quick=args.quick, batch_size=args.batch_size, num_workers=args.num_workers)
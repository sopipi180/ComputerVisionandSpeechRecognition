import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `common` can be found
 
import matplotlib
matplotlib.use("Agg")  # save plots without opening windows
 
from common import config, milk10k
from common.config import TARGET
import part1_metadata
import part2_color
from part3_processing import preprocess_image, process_batch
from part4_loader import MilkDataLoader
from part5_visualizer import show_grid, plot_class_balance, plot_batch_check
 
lines = []
 
 
def log(*text):
    msg = " ".join(str(t) for t in text)
    print(msg)
    lines.append(msg)
 
 
df = milk10k.load_metadata()
available = milk10k.available_subset(df)
log(f"{len(df):,} images in metadata, {len(available):,} found on disk")
 
log("\n===== PART 1: METADATA =====")
part1_metadata.run(df, log)
 
log("\n===== PART 2: COLOUR =====")
part2_color.run(available, log, per_class=25)
 
log("\n===== PART 3: PROCESSING =====")
demo = available.groupby(TARGET).head(2).head(4)
raw, ids, _ = process_batch(demo, size=None, norm=None)
processed, _, _ = process_batch(demo, size=(224, 224), norm="minmax")
gray, _, _ = process_batch(demo, size=(224, 224), color="gray", norm="minmax")
show_grid(list(raw) + list(processed) + list(gray),
          titles=["raw 600x450"] * len(ids) + ["rgb 224x224"] * len(ids) + ["gray 224x224"] * len(ids),
          ncols=len(ids), title="Before (top) and after (rgb, gray)", filename="part3_before_after.png")
_, info = preprocess_image(config.image_path(ids[0]))
log("one processed image:", info)
_, kept, skipped = process_batch([config.image_path(ids[0]), "missing.jpg"])
log("batch with a missing file -> kept:", kept, "skipped:", skipped)
 
log("\n===== PART 4: LOADER =====")
loader = MilkDataLoader(df, batch_size=12)
log(f"{len(loader.df):,} images, {len(loader)} batches, classes: {loader.classes}")
images, labels, ids = next(iter(loader))
log("first batch:", images.shape, "labels:", labels.tolist())
 
log("\n===== PART 5: VISUALISER =====")
show_grid(images, titles=[loader.classes[i] for i in labels],
          title="One batch from the loader", filename="part5_loader_batch.png")
log("3 classes:", plot_class_balance(loader.df, filename="part5_balance_3class.png").to_dict())
log("11 classes:", plot_class_balance(loader.df, "dx11", filename="part5_balance_11class.png").to_dict())
raw4, _, _ = process_batch([config.image_path(i) for i in ids[:4]], size=None, norm=None)
plot_batch_check(images, raw=raw4, filename="part5_batch_check.png")
 
results_dir = Path(__file__).resolve().parent / "results"
results_dir.mkdir(exist_ok=True)
(results_dir / "results.txt").write_text("\n".join(lines))
print("\nDone. Figures in Session2/figures, numbers in Session2/results/results.txt")
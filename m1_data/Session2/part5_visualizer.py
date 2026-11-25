from common import viz
from common.viz import displayable  # noqa: F401  (re-exported for convenience)
from plotting import FIG_DIR
 
 
def show_grid(images, titles=None, ncols=6, title=None, filename="grid.png"):
    return viz.show_grid(images, titles, ncols, title, filename, fig_dir=FIG_DIR)
 
 
def plot_class_balance(df, label_col=viz.TARGET, filename="class_balance.png"):
    return viz.plot_class_balance(df, label_col, filename, fig_dir=FIG_DIR)
 
 
def plot_batch_check(images, raw=None, filename="batch_check.png"):
    return viz.plot_batch_check(images, raw, filename, fig_dir=FIG_DIR)
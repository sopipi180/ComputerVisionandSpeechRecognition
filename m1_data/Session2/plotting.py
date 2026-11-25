from pathlib import Path
 
import matplotlib.pyplot as plt
 
FIG_DIR = Path(__file__).resolve().parent / "figures"
SHOW = False  # True = also pop up each figure
 
# Same colour for each class in every chart
CLASS_COLORS = {"Benign": "#2a78d6", "Malignant": "#eb6834", "Indeterminate": "#1baf7a"}
 
 
def color_of(label):
    return CLASS_COLORS.get(str(label), "#8a8984")
 
 
def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
 
 
def save(fig, name):
    """Save a figure to Session2/figures/, then show or close it."""
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")
    plt.show() if SHOW else plt.close(fig)
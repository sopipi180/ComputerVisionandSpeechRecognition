import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
 
from common.config import TARGET
from plotting import color_of, style, save
 
SKIP = ["isic_id", "lesion_id", TARGET]
 
# Fields that should NOT be used as model features
RISKY = {
    "diagnosis_2": "leakage: part of the diagnosis itself",
    "diagnosis_3": "leakage: part of the diagnosis itself",
    "diagnosis_4": "leakage: part of the diagnosis itself",
    "dx11": "leakage: the 11-class label",
    "melanocytic": "leakage: derived from the diagnosis",
    "diagnosis_confirm_type": "leakage: reflects the doctor's suspicion",
    "concomitant_biopsy": "leakage: biopsies are taken on suspicious lesions",
    "attribution": "bias: proxy for the clinic that took the photo",
    "copyright_license": "bias: proxy for the clinic",
    "image_manipulation": "bias: proxy for device / processing",
    "image_type": "confounder: changes image colours, not the diagnosis",
}
 
 
def column_type(s):
    if s.dropna().empty:
        return "empty"
    if pd.api.types.is_bool_dtype(s) or set(s.dropna().astype(str).str.lower()) <= {"true", "false"}:
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if s.dropna().astype(str).str.len().mean() > 40:
        return "free text"
    return "categorical"
 
 
def per_lesion(df, col):
    """Each lesion has 2 images. If the value is the same for both, count the lesion once."""
    same_for_both = (df.groupby("lesion_id")[col].nunique(dropna=False) <= 1).all()
    return df.drop_duplicates("lesion_id") if same_for_both else df
 
 
def describe_columns(df):
    """Task 1: type and % missing of every metadata column."""
    rows = [{"column": c,
             "type": column_type(df[c]),
             "unique": df[c].nunique(),
             "missing_%": round(100 * df[c].isna().mean(), 1),
             "risk": RISKY.get(c, "")}
            for c in df.columns if c not in SKIP]
    return pd.DataFrame(rows)
 
 
def test_categorical(df, col):
    """Task 2-3: chi-square test + Cramer's V (0 = unrelated, 1 = decides the class)."""
    d = per_lesion(df, col)
    x = d[col].astype(str)
    rare = x.map(x.value_counts()) < 20  # chi-square needs enough counts per group
    table = pd.crosstab(x.where(~rare, "(rare)"), d[TARGET])
    if table.shape[0] < 2:
        return None
    chi2, p, _, _ = stats.chi2_contingency(table)
    v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
    return {"column": col, "test": "chi-square", "effect": v, "p_value": p, "table": table}
 
 
def test_numeric(df, col):
    """Task 3: Kruskal-Wallis (ANOVA on ranks - age isn't normally distributed) + eta squared."""
    d = per_lesion(df, col).dropna(subset=[col])
    groups = [g[col].values for _, g in d.groupby(TARGET)]
    h, p = stats.kruskal(*groups)
    allv = np.concatenate(groups)
    eta2 = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in groups) / ((allv - allv.mean()) ** 2).sum()
    return {"column": col, "test": "Kruskal-Wallis", "effect": eta2, "p_value": p}
 
 
def plot_categorical(res):
    """Stacked bars: % of each class inside each category."""
    table = res["table"]
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index[:12]]
    pct = table.div(table.sum(axis=1), axis=0) * 100
 
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(pct) + 1.5))
    pct.plot(kind="barh", stacked=True, ax=ax, width=0.7, edgecolor="white",
             color=[color_of(c) for c in pct.columns])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of lesions")
    ax.set_ylabel("")
    ax.set_title(f"{res['column']} vs {TARGET}  (Cramer's V = {res['effect']:.2f})", fontsize=10)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.05))
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, f"part1_{res['column']}.png")
 
 
def plot_numeric(df, col, res):
    """Boxplot per class + overlaid histograms."""
    d = per_lesion(df, col).dropna(subset=[col])
    classes = d[TARGET].value_counts().index
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
 
    box = a1.boxplot([d.loc[d[TARGET] == c, col] for c in classes],
                     tick_labels=classes, patch_artist=True)
    for patch, c in zip(box["boxes"], classes):
        patch.set_facecolor(color_of(c))
        patch.set_alpha(0.6)
    a1.set_ylabel(col)
    style(a1)
 
    for c in classes:
        a2.hist(d.loc[d[TARGET] == c, col], bins=17, density=True,
                histtype="step", linewidth=2, color=color_of(c), label=c)
    a2.set_xlabel(col)
    a2.legend(frameon=False)
    style(a2)
 
    fig.suptitle(f"{col} by class   (Kruskal-Wallis p = {res['p_value']:.1e}, "
                 f"eta squared = {res['effect']:.3f})", fontsize=10)
    save(fig, f"part1_{col}.png")
 
 
def plot_overview(results):
    """Task 4: one bar per field, risky ones in red."""
    r = pd.DataFrame(results).sort_values("effect")
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(r) + 1.5))
    ax.barh(r["column"], r["effect"], height=0.6,
            color=["#e34948" if c in RISKY else "#2a78d6" for c in r["column"]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("association with diagnosis_1 (0 = none, 1 = perfect)")
    ax.set_title("Metadata vs diagnosis  -  red = leakage / bias risk", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "part1_overview.png")
 
 
def run(df, log):
    desc = describe_columns(df)
    log(desc.to_string(index=False))
 
    results = []
    for _, row in desc.iterrows():
        col = row["column"]
        if row["type"] in ("categorical", "boolean"):
            res = test_categorical(df, col)
            if res:
                plot_categorical(res)
                results.append(res)
        elif row["type"] == "numeric":
            res = test_numeric(df, col)
            plot_numeric(df, col, res)
            results.append(res)
 
    plot_overview(results)
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "table"} for r in results])
    summary["risk"] = summary["column"].map(RISKY).fillna("")
    summary = summary.sort_values("effect", ascending=False)
    log("\nAssociation with diagnosis_1 (strongest first):")
    log(summary.round(3).to_string(index=False))
    return summary
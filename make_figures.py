"""
Figures and a summary table for the fracture project, built from the CSVs runner_project.py writes.

Put every teammate's results_<name>.csv and efficiency_<name>.csv in one folder, then run:
    python make_figures.py project_outputs
Figures and summary_table.csv are written to project_outputs/figures/.

Needs: pandas, matplotlib, numpy (all already used by the runner).

Figures
  1_threshold_free.png      AUROC and PR-AUC, mean +/- SD across seeds, each seed as a faint dot
  2_threshold_metrics.png   recall, precision, F1, specificity at the validation-tuned threshold
  3_operating_points.png    precision vs. recall for each model, arrow from no augmentation to flip_rotate
  4_efficiency.png          PR-AUC vs. compute, and inference time vs. compute
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- style
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
AUG_COLOR = {"none": "#2a78d6", "flip_rotate": "#eb6834"}          # categorical slots 1 and 2
AUG_LABEL = {"none": "No augmentation", "flip_rotate": "Flip + rotate"}
GROUP_COLOR = {"pretrained": "#2a78d6", "scratch": "#eb6834"}

NAMES = {                                    # display name, group, architecture
    "vgg16_pretrained": ("VGG16, pretrained", "pretrained", "VGG16"),
    "resnet18_pretrained": ("ResNet18, pretrained", "pretrained", "ResNet18"),
    "mobilenetv2_pretrained": ("MobileNetV2, pretrained", "pretrained", "MobileNetV2"),
    "resnet18_scratch": ("ResNet18, scratch", "scratch", "ResNet18"),
    "nin_scratch": ("NiN, scratch", "scratch", "NiN"),
    "majority_class": ("Majority-class baseline", "baseline", "Baseline"),
}
ORDER = list(NAMES)
LABEL_BELOW = {"mobilenetv2_pretrained"}         # efficiency plot: label under the point
METRICS = ["recall", "precision", "f1", "specificity", "auroc", "pr_auc"]
METRIC_LABEL = {"recall": "Recall (sensitivity)", "precision": "Precision", "f1": "F1",
                "specificity": "Specificity", "auroc": "AUROC", "pr_auc": "PR-AUC"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 9, "ytick.labelsize": 9.5, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "legend.fontsize": 9.5,
})


def name(m):
    return NAMES.get(m, (m, "other", m))[0]


def group(m):
    return NAMES.get(m, (m, "other", m))[1]


def model_order(models):
    known = [m for m in ORDER if m in models]
    return known + sorted(m for m in models if m not in ORDER)


# ---------------------------------------------------------------- loading
def load_results(folder):
    files = sorted(glob.glob(os.path.join(folder, "results_*.csv")))
    if not files:
        sys.exit(f"No results_*.csv files found in '{folder}'.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["weighted"] = df["weighted"].astype(str).str.lower().isin(["true", "1"])
    before = len(df)
    df = df.drop_duplicates(subset=["model", "aug", "seed", "weighted"], keep="first")
    if len(df) < before:
        print(f"Dropped {before - len(df)} duplicate run(s) (same model, aug, seed and weighting).")
    return df


def summarize(raw):
    """Mean, SD and seed count per (model, aug), unweighted runs only."""
    d = raw[~raw["weighted"]]
    cols = [c for c in METRICS if c in d.columns]
    g = d.groupby(["model", "aug"])[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(g.size().rename("n"))
    return out.reset_index()


def load_efficiency(folder):
    files = sorted(glob.glob(os.path.join(folder, "efficiency_*.csv")))
    if not files:
        return None
    eff = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "runner" in eff and eff["runner"].nunique() > 1:
        print("Note: inference times come from more than one person's machine "
              f"({', '.join(eff['runner'].unique())}). They are only comparable if the hardware was the same.")
    return eff.drop_duplicates(subset="model", keep="first").reset_index(drop=True)


# ---------------------------------------------------------------- figures
def _dot_panel(ax, summ, raw, metric, models, augs, ref=None, ref_label=None):
    ys = np.arange(len(models))[::-1]
    offs = {a: o for a, o in zip(augs, np.linspace(-0.17, 0.17, len(augs)))} if len(augs) > 1 else {augs[0]: 0}
    for m, y in zip(models, ys):
        rows = summ[summ.model == m]
        for _, r in rows.iterrows():
            if r["aug"] == "-":                         # baseline: one hollow marker
                ax.plot(r[f"{metric}_mean"], y, "o", ms=8, mfc=SURFACE, mec=MUTED, mew=1.6, zorder=3)
                continue
            yy, col = y + offs.get(r["aug"], 0), AUG_COLOR.get(r["aug"], MUTED)
            if raw is not None:
                seeds = raw[(raw.model == m) & (raw.aug == r["aug"]) & (~raw.weighted)][metric]
                ax.plot(seeds, [yy] * len(seeds), "o", ms=4.5, color=col, alpha=0.35, mew=0, zorder=2)
            sd = r[f"{metric}_std"]
            ax.errorbar(r[f"{metric}_mean"], yy, xerr=0 if pd.isna(sd) else sd, fmt="o", ms=8,
                        color=col, mec=SURFACE, mew=1.5, elinewidth=2, capsize=0, zorder=3)
    if ref is not None:
        ax.axvline(ref, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        if ref_label:
            ax.annotate(ref_label, (ref, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -2),
                        textcoords="offset points", va="top", fontsize=8.5, color=INK2)
    ax.set_yticks(ys, [name(m) for m in models])
    ax.grid(axis="y", visible=False)
    ax.set_title(METRIC_LABEL[metric])
    for gy in ys[:-1]:
        ax.axhline(gy - 0.5, color=GRID, lw=0.8, zorder=0)


def _aug_legend(fig, augs, raw):
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=AUG_COLOR[a], mec=SURFACE) for a in augs]
    labels = [AUG_LABEL[a] for a in augs]
    note = "Dot and line: mean ± 1 SD across seeds" + ("; faint dots: individual seeds" if raw is not None else "")
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.995), ncol=len(augs) + 1, handletextpad=0.3)
    fig.text(0.99, 0.975, note, ha="right", va="center", fontsize=8.5, color=INK2)


def fig_threshold_free(summ, raw, out):
    models = model_order(set(summ.model))
    augs = [a for a in AUG_COLOR if a in set(summ.aug)]
    prev = summ.loc[summ.model == "majority_class", "pr_auc_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(models) + 1.6), sharey=True)
    _dot_panel(axes[0], summ, raw, "auroc", models, augs, ref=0.5, ref_label="chance")
    _dot_panel(axes[1], summ, raw, "pr_auc", models, augs,
               ref=float(prev.iloc[0]) if len(prev) else None,
               ref_label=f"chance = {float(prev.iloc[0]):.3f}" if len(prev) else None)
    _aug_legend(fig, augs, raw)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_threshold_metrics(summ, raw, out):
    models = model_order(set(summ.model) - {"majority_class"})
    augs = [a for a in AUG_COLOR if a in set(summ.aug)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 1.05 * len(models) + 3), sharey=True)
    for ax, metric in zip(axes.flat, ["recall", "precision", "f1", "specificity"]):
        _dot_panel(ax, summ, raw, metric, models, augs)
    _aug_legend(fig, augs, raw)
    fig.text(0.01, 0.005, "All four use the threshold that maximized F1 on the validation set, so they move together "
             "when that threshold moves.", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_operating_points(summ, out):
    s = summ[summ.model != "majority_class"]
    fig, ax = plt.subplots(figsize=(8, 6.4))
    rr = np.linspace(0.01, 1, 400)
    for f in (0.4, 0.5, 0.6, 0.7):                     # iso-F1 curves: precision = f*r / (2r - f)
        pp = f * rr / (2 * rr - f)
        ok = (2 * rr - f > 0) & (pp <= 1)
        ax.plot(rr[ok], pp[ok], color=GRID, lw=1.2, zorder=0)
        i = np.argmin(np.abs(rr - 0.9)) if f < 0.9 else 0
        ax.annotate(f"F1 = {f}", (rr[i], pp[i]), xytext=(0, 3), textcoords="offset points",
                    fontsize=8.5, color=MUTED, ha="center")
    for m in model_order(set(s.model)):
        rows = s[s.model == m].set_index("aug")
        if {"none", "flip_rotate"} <= set(rows.index):
            a, b = rows.loc["none"], rows.loc["flip_rotate"]
            ax.annotate("", xy=(b.recall_mean, b.precision_mean), xytext=(a.recall_mean, a.precision_mean),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3, shrinkA=6, shrinkB=6), zorder=2)
        for aug, r in rows.iterrows():
            ax.plot(r.recall_mean, r.precision_mean, "o", ms=9, color=AUG_COLOR.get(aug, MUTED),
                    mec=SURFACE, mew=1.5, zorder=3)
        anchor = rows.loc["none"] if "none" in rows.index else rows.iloc[0]
        ax.annotate(name(m), (anchor.recall_mean, anchor.precision_mean), xytext=(9, -3),
                    textcoords="offset points", fontsize=9.5, color=INK)
    ax.set_xlabel("Recall: share of fractures caught")
    ax.set_ylabel("Precision: share of fracture calls that were right")
    lo_r, hi_r = s.recall_mean.min(), s.recall_mean.max()
    lo_p, hi_p = s.precision_mean.min(), s.precision_mean.max()
    ax.set_xlim(max(0, lo_r - 0.06), min(1, hi_r + 0.16))
    ax.set_ylim(max(0, lo_p - 0.06), min(1, hi_p + 0.06))
    ax.set_title("Where each model operates, and how augmentation moved it")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=AUG_COLOR[a], mec=SURFACE) for a in AUG_COLOR]
    ax.legend(handles, [AUG_LABEL[a] for a in AUG_COLOR], loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_efficiency(summ, eff, out):
    e = eff.copy()
    e["arch"] = [NAMES.get(m, (m, "", m))[2] for m in e.model]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    for _, r in e.iterrows():
        rows = summ[summ.model == r.model]
        for _, s in rows.iterrows():
            ax.plot(r.gflops, s.pr_auc_mean, "o", ms=9, color=AUG_COLOR.get(s.aug, MUTED), mec=SURFACE, mew=1.5)
        if len(rows):
            below = r.model in LABEL_BELOW                 # avoids collisions between nearby labels
            y = rows.pr_auc_mean.min() if below else rows.pr_auc_mean.max()
            ax.annotate(name(r.model), (r.gflops, y), xytext=(8, -14 if below else 4),
                        textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("GFLOPs per image (log scale)")
    ax.set_title("PR-AUC vs. compute")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, color=AUG_COLOR[a], mec=SURFACE) for a in AUG_COLOR]
    ax.legend(handles, [AUG_LABEL[a] for a in AUG_COLOR], loc="lower right")
    ax.margins(x=0.25, y=0.15)

    ax = axes[1]
    arch = e.groupby("arch").agg(gflops=("gflops", "first"), ms=("inference_ms", "mean"),
                                 params=("params_millions", "first")).reset_index()
    for _, r in arch.iterrows():
        ax.plot(r.gflops, r.ms, "o", ms=9, color=INK2, mec=SURFACE, mew=1.5)
        ax.annotate(f"{r.arch} ({r.params:.1f}M params)", (r.gflops, r.ms), xytext=(8, 4),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("GFLOPs per image (log scale)")
    ax.set_ylabel("Milliseconds per image")
    ax.set_title("Inference time vs. compute")
    ax.margins(x=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def write_table(summ, eff, out):
    rows = []
    for m in model_order(set(summ.model)):
        for _, r in summ[summ.model == m].iterrows():
            row = {"model": name(m), "augmentation": AUG_LABEL.get(r.aug, r.aug), "seeds": int(r.n)}
            for c in METRICS:
                if f"{c}_mean" in r:
                    sd = r[f"{c}_std"]
                    row[METRIC_LABEL[c]] = f"{r[f'{c}_mean']:.3f}" + ("" if pd.isna(sd) else f" ± {sd:.3f}")
            if eff is not None and m in set(eff.model):
                er = eff[eff.model == m].iloc[0]
                row.update({"params (M)": round(er.params_millions, 2), "GFLOPs": round(er.gflops, 2),
                            "inference (ms)": round(er.inference_ms, 2)})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)


def make_all(summ, raw, eff, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig_threshold_free(summ, raw, os.path.join(outdir, "1_threshold_free.png"))
    fig_threshold_metrics(summ, raw, os.path.join(outdir, "2_threshold_metrics.png"))
    fig_operating_points(summ, os.path.join(outdir, "3_operating_points.png"))
    if eff is not None:
        fig_efficiency(summ, eff, os.path.join(outdir, "4_efficiency.png"))
    write_table(summ, eff, os.path.join(outdir, "summary_table.csv"))
    print(f"Figures and summary_table.csv written to {outdir}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "project_outputs"
    raw = load_results(folder)
    make_all(summarize(raw), raw, load_efficiency(folder), os.path.join(folder, "figures"))

#!/usr/bin/env python
import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import wilcoxon, pearsonr, entropy

OUT_DIR = "out"
FIG_NAME = "results_xai_f1_lines.pdf"

DATASETS = ["kidney", "covid"]
MODELS = ["siglip2", "biomedclip", "radimagenet"]

WEIGHTS_FULL = ["medical", "generic"]
WEIGHT_EXP = "generic"

XAI_LEVELS = [0, 10, 25, 50]
ALPHAS = [0.0, 0.5, 1.0]
ALPHA_TEST = 0.05

PALETTE = {
    "siglip2": "#2ca02c",
    "biomedclip": "#7b2cbf",
    "radimagenet": "#1f77b4",
}

MODEL_NAMES = {
    "radimagenet": "RadImageNet",
    "biomedclip": "BiomedCLIP",
    "siglip2": "SigLIP2",
}

sns.set_theme(style="whitegrid", context="talk")

fontsize = 6


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def metrics_global(y_true, probs):
    y_pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    pre, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return acc, pre, rec, f1


def summarize_files(files):
    fold_metrics = []

    for path in files:
        d = load_pkl(path)
        fold_metrics.append(metrics_global(d["y_true"], d["probs"]))

    if not fold_metrics:
        return None

    arr = np.array(fold_metrics)

    return {
        "n_folds": len(fold_metrics),
        "acc": arr[:, 0].mean(),
        "precision": arr[:, 1].mean(),
        "recall": arr[:, 2].mean(),
        "f1": arr[:, 3].mean(),
        "f1_std": arr[:, 3].std(),
        "f1_folds": arr[:, 3],
    }


def get_full_files(dataset, model, weight):
    prefix = f"{dataset}_{model}_{weight}_full"
    return sorted(glob.glob(os.path.join(OUT_DIR, f"{prefix}_k*.pkl")))


# ============================================================
# EXPERIMENT: FULL
# ============================================================

rows_full = []

for dataset in DATASETS:
    for model in MODELS:
        for weight in WEIGHTS_FULL:
            files = get_full_files(dataset, model, weight)
            m = summarize_files(files)

            if m is None:
                print(f"No FULL files found for: {dataset}_{model}_{weight}")
                continue

            rows_full.append({
                "dataset": dataset,
                "model": MODEL_NAMES[model],
                "weights": weight,
                "n_folds": m["n_folds"],
                "acc": m["acc"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            })

df_full = pd.DataFrame(rows_full).round(4)

print("\nEXPERIMENT: FULL")
print(df_full.to_string(index=False))


# ============================================================
# EXPERIMENT: WILCOXON FULL
# ============================================================

rows_wilcoxon = []

for dataset in DATASETS:
    for model in MODELS:
        m_med = summarize_files(get_full_files(dataset, model, "medical"))
        m_gen = summarize_files(get_full_files(dataset, model, "generic"))

        if m_med is None or m_gen is None:
            continue

        n = min(len(m_med["f1_folds"]), len(m_gen["f1_folds"]))

        if n < 2:
            continue

        f1_med = m_med["f1_folds"][:n]
        f1_gen = m_gen["f1_folds"][:n]

        try:
            stat, p = wilcoxon(f1_med, f1_gen)
        except ValueError:
            continue

        if p < ALPHA_TEST:
            rows_wilcoxon.append({
                "dataset": dataset,
                "model": MODEL_NAMES[model],
                "n_folds": n,
                "f1_medical": f1_med.mean(),
                "f1_generic": f1_gen.mean(),
                "diff_generic-medical": f1_gen.mean() - f1_med.mean(),
                "p_value": p,
            })

df_wilcoxon = pd.DataFrame(rows_wilcoxon).round(4)

print("\nEXPERIMENT: WILCOXON FULL")
if df_wilcoxon.empty:
    print("No significant differences.")
else:
    print(df_wilcoxon.to_string(index=False))


# ============================================================
# EXPERIMENT: XAI
# ============================================================

rows_xai = []

for dataset in DATASETS:
    for model in MODELS:
        for level in XAI_LEVELS:
            prefix = f"{dataset}_{model}_{WEIGHT_EXP}_xai{level}"
            files = sorted(glob.glob(os.path.join(OUT_DIR, f"{prefix}_k*.pkl")))
            m = summarize_files(files)

            if m is None:
                print(f"No XAI files found for: {prefix}")
                continue

            rows_xai.append({
                "dataset": dataset,
                "model": model,
                "weights": WEIGHT_EXP,
                "xai_level": level,
                "n_folds": m["n_folds"],
                "acc": m["acc"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "f1_std": m["f1_std"],
            })

df_xai = pd.DataFrame(rows_xai).round(4)

print("\nEXPERIMENT: XAI")
if df_xai.empty:
    print("Empty DataFrame")
else:
    df_xai_print = df_xai.copy()
    df_xai_print["model"] = df_xai_print["model"].map(MODEL_NAMES)
    print(df_xai_print[[
        "dataset",
        "model",
        "weights",
        "xai_level",
        "n_folds",
        "acc",
        "precision",
        "recall",
        "f1",
    ]].to_string(index=False))


# ============================================================
# EXPERIMENT: XAI FIGURE
# ============================================================

if not df_xai.empty:
    fig, axes = plt.subplots(
        1, 2,
        figsize=(6.8, 3.2),
        sharey=True
    )

    handles = []
    labels = []

    for ax, dataset in zip(axes, DATASETS):
        df_d = df_xai[df_xai["dataset"] == dataset]

        for model in MODELS:
            df_m = df_d[df_d["model"] == model].sort_values("xai_level")

            if df_m.empty:
                continue

            x = df_m["xai_level"].to_numpy(dtype=float)
            y = df_m["f1"].to_numpy(dtype=float)
            yerr = df_m["f1_std"].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.4,
                markersize=6,
                color=PALETTE[model],
                label=MODEL_NAMES[model],
            )

            ax.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=PALETTE[model],
                alpha=0.18,
                linewidth=0,
            )

        ax.set_title(dataset.upper(), fontweight="bold", fontsize=fontsize*2)
        ax.set_xlabel("Masked regions (%)", fontsize=fontsize)
        ax.set_xticks(XAI_LEVELS)
        ax.set_xticklabels([str(v) for v in XAI_LEVELS])
        ax.tick_params(axis="both", labelsize=fontsize*1.5)
        ax.grid(True)

        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    axes[0].set_ylabel("Weighted F1", fontsize=fontsize*1.5)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
        fontsize=fontsize,
    )

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(FIG_NAME, bbox_inches="tight")
    plt.show()


# ============================================================
# EXPERIMENT: SSL
# ============================================================

rows_ssl = []
rows_ssl_pseudo = []

for dataset in DATASETS:
    for model in MODELS:
        for alpha in ALPHAS:
            alpha_str = str(alpha).replace(".", "")
            prefix = f"{dataset}_{model}_{WEIGHT_EXP}_hybrid_alpha{alpha_str}"
            files = sorted(glob.glob(os.path.join(OUT_DIR, f"{prefix}_k*.pkl")))
            m = summarize_files(files)

            if m is None:
                print(f"No SSL files found for: {prefix}")
                continue

            pseudo_accs = []
            pseudo_pres = []
            pseudo_recs = []
            pseudo_f1s = []
            pseudo_n = []

            for path in files:
                d = load_pkl(path)

                if "y_xu_true" in d and "y_xu_pseudo" in d:
                    xu_acc = accuracy_score(d["y_xu_true"], d["y_xu_pseudo"])
                    xu_pre, xu_rec, xu_f1, _ = precision_recall_fscore_support(
                        d["y_xu_true"],
                        d["y_xu_pseudo"],
                        average="weighted",
                        zero_division=0,
                    )

                    pseudo_accs.append(xu_acc)
                    pseudo_pres.append(xu_pre)
                    pseudo_recs.append(xu_rec)
                    pseudo_f1s.append(xu_f1)
                    pseudo_n.append(len(d["y_xu_true"]))

            rows_ssl.append({
                "dataset": dataset,
                "model": MODEL_NAMES[model],
                "weights": WEIGHT_EXP,
                "alpha": alpha,
                "n_folds": m["n_folds"],
                "acc": m["acc"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            })

            rows_ssl_pseudo.append({
                "dataset": dataset,
                "model": MODEL_NAMES[model],
                "weights": WEIGHT_EXP,
                "alpha": alpha,
                "n_folds": m["n_folds"],
                "pseudo_acc": np.mean(pseudo_accs) if pseudo_accs else np.nan,
                "pseudo_precision": np.mean(pseudo_pres) if pseudo_pres else np.nan,
                "pseudo_recall": np.mean(pseudo_recs) if pseudo_recs else np.nan,
                "pseudo_f1": np.mean(pseudo_f1s) if pseudo_f1s else np.nan,
                "n_xu": int(np.mean(pseudo_n)) if pseudo_n else 0,
            })

df_ssl = pd.DataFrame(rows_ssl).round(4)
df_ssl_pseudo = pd.DataFrame(rows_ssl_pseudo).round(4)

print("\nEXPERIMENT: SSL")
print(df_ssl[[
    "dataset",
    "model",
    "alpha",
    "n_folds",
    "acc",
    "precision",
    "recall",
    "f1",
]].to_string(index=False))

print("\nEXPERIMENT: SSL PSEUDO-LABEL QUALITY")
print(df_ssl_pseudo[[
    "dataset",
    "model",
    "alpha",
    "n_folds",
    "pseudo_acc",
    "pseudo_precision",
    "pseudo_recall",
    "pseudo_f1",
    "n_xu",
]].to_string(index=False))


# ============================================================
# EXPERIMENT: XAI VS UNCERTAINTY
# ============================================================

rows_xai_corr = []

for dataset in DATASETS:
    for model in MODELS:
        for alpha in ALPHAS:
            alpha_str = str(alpha).replace(".", "")
            prefix = f"{dataset}_{model}_{WEIGHT_EXP}_hybrid_alpha{alpha_str}"
            files = sorted(glob.glob(os.path.join(OUT_DIR, f"{prefix}_k*.pkl")))

            if len(files) == 0:
                continue

            rs = []
            accs = []
            pres = []
            recs = []
            f1s = []

            for path in files:
                d = load_pkl(path)

                if not all(k in d for k in [
                    "y_xu_probs",
                    "y_xu_true",
                    "y_xu_pseudo",
                    "y_xai_scores",
                ]):
                    continue

                probs = d["y_xu_probs"]

                ent = entropy(probs.T, base=2)
                ent /= np.log2(probs.shape[1])

                shap = d["y_xai_scores"]

                r, _ = pearsonr(ent, shap)
                rs.append(r)

                acc = accuracy_score(d["y_xu_true"], d["y_xu_pseudo"])
                pre, rec, f1, _ = precision_recall_fscore_support(
                    d["y_xu_true"],
                    d["y_xu_pseudo"],
                    average="weighted",
                    zero_division=0,
                )

                accs.append(acc)
                pres.append(pre)
                recs.append(rec)
                f1s.append(f1)

            if len(rs) == 0:
                continue

            rows_xai_corr.append({
                "dataset": dataset,
                "model": MODEL_NAMES[model],
                "alpha": alpha,
                "n_folds": len(rs),
                "pearson": np.mean(rs),
                "acc": np.mean(accs),
                "precision": np.mean(pres),
                "recall": np.mean(recs),
                "f1": np.mean(f1s),
            })

df_xai_corr = pd.DataFrame(rows_xai_corr).round(4)

print("\nEXPERIMENT: XAI VS UNCERTAINTY")
print(df_xai_corr.to_string(index=False))


# ============================================================
# EXPERIMENT: XAI MASKING EXAMPLES
# ============================================================

PDF_MASKING = "results_xai_masking_examples.pdf"

EXAMPLE_DATASETS = ["kidney"]
EXAMPLE_FOLDS = [1]
EXAMPLE_LEVELS = [0, 10, 25, 50]

figs_to_show = []

with PdfPages(PDF_MASKING) as pdf:
    for dataset in EXAMPLE_DATASETS:
        for fold in EXAMPLE_FOLDS:
            paths = [
                os.path.join(
                    OUT_DIR,
                    f"{dataset}_k{fold}_shap{level}.png"
                )
                for level in EXAMPLE_LEVELS
            ]

            if not all(os.path.exists(p) for p in paths):
                print(f"No masking images found for {dataset} k{fold}")
                continue

            fig, axes = plt.subplots(1, len(EXAMPLE_LEVELS), figsize=(6, 3))

            for ax, level, path in zip(axes, EXAMPLE_LEVELS, paths):
                img = mpimg.imread(path)
                ax.imshow(img)
                ax.set_title("Original" if level == 0 else f"Mask {level}%", fontsize=fontsize)
                ax.axis("off")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            figs_to_show.append(fig)

print(f"Saved: {PDF_MASKING}")

#for fig in figs_to_show:
#    fig.show()

plt.show()
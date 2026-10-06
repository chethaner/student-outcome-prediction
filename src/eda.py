"""Exploratory data analysis: text overview + figures (permitted enrollment-time features only)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .data_utils import (BINARY_FEATURES, FIG_DIR, NOMINAL_FEATURES, NUMERIC_FEATURES,
                         OUTPUT_DIR, TARGET, prohibited_columns)

CLASS_COLORS = {"Dropout": "#d1495b", "Enrolled": "#edae49", "Graduate": "#2e8b57"}


def data_overview(df: pd.DataFrame) -> str:
    L = []
    L.append(f"Rows: {df.shape[0]}\nColumns: {df.shape[1]} (36 inputs + Target expected)")
    L.append("\nColumn dtypes:\n" + df.dtypes.astype(str).to_string())
    vc = df[TARGET].value_counts()
    L.append("\nTarget classes:\n" + pd.DataFrame(
        {"count": vc, "share_%": (vc / len(df) * 100).round(2)}).to_string())
    miss = df.isna().sum()
    L.append(f"\nMissing values (total): {int(miss.sum())}")
    if miss.sum():
        L.append(miss[miss > 0].to_string())
    L.append(f"Duplicate rows: {int(df.duplicated().sum())}")
    proh = prohibited_columns(df)
    L.append(f"\nProhibited semester-performance columns found: {len(proh)} (expected 12) "
             "-> excluded from primary model:\n  - " + "\n  - ".join(proh))
    L.append("\nPermitted numeric features:\n" + df[NUMERIC_FEATURES].describe().T.round(2).to_string())
    L.append("\nDistinct values per permitted binary/categorical feature:\n"
             + df[BINARY_FEATURES + NOMINAL_FEATURES].nunique().to_string())
    L.append("\nData-quality checks:")
    for c in BINARY_FEATURES:
        bad = (~df[c].isin([0, 1])).sum()
        if bad:
            L.append(f"  - {c}: {bad} values outside {{0,1}}")
    for c in ["Previous qualification (grade)", "Admission grade"]:
        L.append(f"  - {c}: min={df[c].min()}, max={df[c].max()}, zeros={(df[c] == 0).sum()}")
    L.append(f"  - Age at enrollment: min={df['Age at enrollment'].min()}, max={df['Age at enrollment'].max()}")
    for c in NOMINAL_FEATURES:
        rare = (df[c].value_counts() < 10).sum()
        L.append(f"  - {c}: {df[c].nunique()} levels, {rare} levels with <10 rows (pooled by the encoder)")
    return "\n".join(L)


def _share_bars(ax, df, col, top=None):
    levels = df[col].value_counts().index[:top] if top else df[col].unique()
    ct = pd.crosstab(df.loc[df[col].isin(levels), col], df[TARGET], normalize="index")
    ct.plot(kind="bar", stacked=True, ax=ax, legend=False,
            color=[CLASS_COLORS.get(c, "grey") for c in ct.columns], width=0.85)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("share of students")
    ax.tick_params(axis="x", labelsize=7)


def run_eda(df: pd.DataFrame, fig_dir=FIG_DIR, out_dir=OUTPUT_DIR) -> str:
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = data_overview(df)
    (out_dir / "data_overview.txt").write_text(text, encoding="utf-8")
    classes = sorted(df[TARGET].unique())

    # 1. target distribution
    fig, ax = plt.subplots(figsize=(5, 3.5))
    vc = df[TARGET].value_counts().reindex(classes)
    bars = ax.bar(vc.index, vc.values, color=[CLASS_COLORS.get(c, "grey") for c in vc.index])
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / len(df):.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_title("Target distribution"); ax.set_ylim(0, vc.max() * 1.2)
    fig.tight_layout(); fig.savefig(fig_dir / "eda_target_distribution.png", dpi=130); plt.close(fig)

    # 2. numeric features by target
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, c in zip(axes.ravel(), NUMERIC_FEATURES):
        ax.boxplot([df.loc[df[TARGET] == k, c] for k in classes], showfliers=False)
        ax.set_xticks(range(1, len(classes) + 1)); ax.set_xticklabels(classes)
        ax.set_title(c, fontsize=9)
    for ax in axes.ravel()[len(NUMERIC_FEATURES):]:
        ax.axis("off")
    fig.suptitle("Numeric enrollment-time features vs Target"); fig.tight_layout()
    fig.savefig(fig_dir / "eda_numeric_vs_target.png", dpi=130); plt.close(fig)

    # 3. binary features vs target
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, c in zip(axes.ravel(), BINARY_FEATURES):
        _share_bars(ax, df, c)
    axes.ravel()[0].legend(classes, fontsize=7)
    fig.suptitle("Binary features: outcome share per value"); fig.tight_layout()
    fig.savefig(fig_dir / "eda_binary_vs_target.png", dpi=130); plt.close(fig)

    # 4. main nominal features vs target (top levels)
    cols = ["Course", "Application mode", "Marital status", "Previous qualification"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, c in zip(axes.ravel(), cols):
        _share_bars(ax, df, c, top=12)
    axes.ravel()[0].legend(classes, fontsize=7)
    fig.suptitle("Categorical features (12 most frequent codes): outcome share"); fig.tight_layout()
    fig.savefig(fig_dir / "eda_categorical_vs_target.png", dpi=130); plt.close(fig)

    # 5. correlation of permitted numeric features
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[NUMERIC_FEATURES].corr()
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im); ax.set_title("Correlation (permitted numeric features)")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_numeric_correlation.png", dpi=130); plt.close(fig)
    return text


if __name__ == "__main__":
    from .data_utils import load_data
    print(run_eda(load_data()))

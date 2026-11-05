"""TURBINE-X EDA Visualization Library.
Generates publication-quality charts for distributions, correlations, and annual trends.
"""
from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Optional[Path] = None, title: str = "Feature Correlation Matrix"):
    """Plot high-contrast correlation heatmap for gas turbine variables."""
    cols = [c for c in df.columns if c != "source_year"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_feature_distributions(df: pd.DataFrame, columns: Optional[List[str]] = None, output_path: Optional[Path] = None):
    """Plot multi-panel histogram and KDE grid for features."""
    cols = columns or [c for c in df.columns if c != "source_year"][:9]
    n_cols = 3
    n_rows = int(np.ceil(len(cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows), dpi=150)
    axes = np.array(axes).ravel()

    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#1f77b4", bins=35, stat="density", alpha=0.6)
        axes[i].set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")

    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_annual_emission_trends(df: pd.DataFrame, output_path: Optional[Path] = None):
    """Plot emission levels (CO and NOx) across operational years (2011-2015)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    sns.boxplot(x="source_year", y="CO", data=df, ax=ax1, palette="Blues")
    ax1.set_title("CO Emission Distribution Across Years", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("CO (mg/m³)")
    ax1.set_yscale("log")

    sns.boxplot(x="source_year", y="NOX", data=df, ax=ax2, palette="Oranges")
    ax2.set_title("NOx Emission Distribution Across Years", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("NOx (mg/m³)")

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


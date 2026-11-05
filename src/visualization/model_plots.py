"""TURBINE-X Model Evaluation & Diagnostic Plots.
Publication-quality plots for actual-vs-predicted, residuals, benchmark comparisons, and importance.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    model_name: str = "Model",
    output_path: Optional[Path] = None
):
    """Plot scatter plot of actual vs predicted with ideal 1:1 diagonal."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()

    # Subsample if large for crisp visualization
    if len(yt) > 3000:
        idx = np.random.choice(len(yt), 3000, replace=False)
        yt_plot, yp_plot = yt[idx], yp[idx]
    else:
        yt_plot, yp_plot = yt, yp

    ax.scatter(yt_plot, yp_plot, alpha=0.3, color="#2b5c8f", s=18, edgecolors="none")

    # 1:1 reference line
    min_val = min(float(np.min(yt)), float(np.min(yp)))
    max_val = max(float(np.max(yt)), float(np.max(yp)))
    ax.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=1.8, label="Ideal 1:1 Line")

    ax.set_title(f"{model_name}: Actual vs Predicted ({target_name})", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.legend(frameon=True)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    model_name: str = "Model",
    output_path: Optional[Path] = None
):
    """Plot dual-panel residual diagnostics: residual vs predicted and residual histogram."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    res = yt - yp

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Residuals vs Predicted
    if len(res) > 3000:
        idx = np.random.choice(len(res), 3000, replace=False)
        yp_plot, res_plot = yp[idx], res[idx]
    else:
        yp_plot, res_plot = yp, res

    ax1.scatter(yp_plot, res_plot, alpha=0.3, color="#34495e", s=18, edgecolors="none")
    ax1.axhline(0, color="#e74c3c", linestyle="--", linewidth=1.5)
    ax1.set_title(f"Residuals vs Predicted ({target_name})", fontsize=11, fontweight="bold")
    ax1.set_xlabel(f"Predicted {target_name}")
    ax1.set_ylabel("Residual (Actual - Predicted)")

    # 2. Residual Distribution
    sns.histplot(res, kde=True, ax=ax2, color="#16a085", bins=40, stat="density")
    ax2.axvline(0, color="#e74c3c", linestyle="--", linewidth=1.5)
    ax2.set_title(f"Residual Distribution ({target_name})", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Residual Error")
    ax2.set_ylabel("Density")

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_model_comparison_bar(
    metrics_df: pd.DataFrame,
    metric_name: str = "RMSE",
    output_path: Optional[Path] = None
):
    """Plot grouped bar chart comparing models on a specific metric across targets."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    sns.barplot(
        data=metrics_df,
        x="model",
        y=metric_name,
        hue="target",
        ax=ax,
        palette="tab10"
    )

    ax.set_title(f"Model Comparison: {metric_name} Across Targets", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Model Architecture", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    plt.xticks(rotation=30, ha="right")
    ax.legend(title="Target", frameon=True)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_feature_importance_bar(
    df_imp: pd.DataFrame,
    title: str = "Feature Importance",
    top_n: int = 12,
    output_path: Optional[Path] = None
):
    """Plot horizontal bar chart of top feature importances."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    top_df = df_imp.head(top_n).sort_values("importance", ascending=True)

    ax.barh(top_df["feature"], top_df["importance"], color="#2980b9", edgecolor="none", alpha=0.85)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Metric")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


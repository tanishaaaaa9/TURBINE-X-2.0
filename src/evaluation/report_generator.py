"""TURBINE-X Report Generator Module.
Generates automated markdown and tabular reports from benchmark evaluations.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def generate_benchmark_markdown_report(metrics_df: pd.DataFrame, output_path: Optional[Path] = None) -> str:
    """Format benchmark evaluation metrics into clean GitHub-flavored markdown tables."""
    lines = [
        "# TURBINE-X Empirical Benchmark Leaderboard",
        "",
        "Evaluation performed strictly on the **Locked 2015 Chronological Test Partition** ($N = 7,384$).",
        ""
    ]

    for track, group in metrics_df.groupby("track"):
        lines.append(f"## {track.replace('_', ' ').title()}")
        lines.append("")
        disp_cols = ["model", "target", "RMSE", "MAE", "R2", "MedAE", "train_time_sec", "latency_ms_per_row"]
        avail = [c for c in disp_cols if c in group.columns]
        sorted_group = group[avail].sort_values(["target", "RMSE"])
        lines.append(sorted_group.to_markdown(index=False))
        lines.append("")

    report_content = "\n".join(lines)
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report_content, encoding="utf-8")

    return report_content


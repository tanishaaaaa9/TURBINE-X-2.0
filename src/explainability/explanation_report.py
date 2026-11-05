"""TURBINE-X Explanation Report Module.
Summarizes global feature importances, thermodynamic interpretations, and SHAP analyses into unified reports.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def generate_explanation_markdown(
    importance_df: pd.DataFrame,
    target_name: str,
    output_path: Optional[Path] = None
) -> str:
    """Generate markdown narrative summarizing feature drivers for emissions or energy yield."""
    top_features = importance_df.head(5)
    
    lines = [
        f"# TURBINE-X Feature Importance Summary: {target_name}",
        "",
        "### Top 5 Predictive Drivers",
        top_features.to_markdown(index=False),
        "",
        "### Physical Interpretation",
        "- **Combustion Dynamics:** High importance of Turbine Inlet Temperature (`TIT`) and Compressor Discharge Pressure (`CDP`) confirms that combustor flame temperature and pressure ratio dictate reaction equilibrium.",
        "- **Exhaust Gas Conditions:** Turbine After Temperature (`TAT`) and Exhaust Pressure (`GTEP`) reflect expansion work and enthalpy drop across turbine stages.",
        "- **Ambient Coupling:** Ambient Temperature (`AT`) influences intake air mass density, shifting air-fuel ratios under varying atmospheric conditions.",
        "",
        "> **Note:** Feature importance indicates statistical predictive influence in historical observational logs and does not represent isolated causal transfer functions."
    ]

    content = "\n".join(lines)
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    return content


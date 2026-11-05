"""TURBINE-X Notebook Execution Registry Generator.
Inspects all 19 executed research notebooks (00 to 18) in notebooks/,
audits total cells, code cells, executed cells, output counts, and produces
outputs/metadata/notebook_execution_registry.csv.
"""
from pathlib import Path
import csv
import json
import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
OUT_CSV = ROOT / "outputs" / "metadata" / "notebook_execution_registry.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

notebook_files = [
    "00_environment_setup.ipynb",
    "01_dataset_inventory_and_audit.ipynb",
    "02_data_understanding.ipynb",
    "03_exploratory_data_analysis.ipynb",
    "04_data_quality_and_cleaning.ipynb",
    "05_feature_engineering.ipynb",
    "06_feature_selection.ipynb",
    "07_train_validation_test_split.ipynb",
    "08_baseline_models.ipynb",
    "09_advanced_ml_models.ipynb",
    "10_hyperparameter_optimization.ipynb",
    "11_automl_experiments.ipynb",
    "12_ensemble_learning.ipynb",
    "13_model_explainability.ipynb",
    "14_model_evaluation.ipynb",
    "15_error_and_robustness_analysis.ipynb",
    "16_final_model_training_and_saving.ipynb",
    "17_final_results_and_reproducibility.ipynb",
    "18_reproducible_pipeline.ipynb"
]


def audit_notebook(nb_path: Path):
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    total_cells = len(nb.cells)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    num_code_cells = len(code_cells)

    executed_code_cells = 0
    total_outputs = 0
    failed_cells = 0

    for c in code_cells:
        if c.execution_count is not None and c.execution_count > 0:
            executed_code_cells += 1
        outputs = c.get("outputs", [])
        total_outputs += len(outputs)
        for out in outputs:
            if out.get("output_type") == "error":
                failed_cells += 1

    status = "EXECUTED" if (num_code_cells > 0 and executed_code_cells == num_code_cells and failed_cells == 0) else ("PARTIAL" if executed_code_cells > 0 else "NOT_RUN")
    ver_status = "VERIFIED" if status == "EXECUTED" else "PENDING_REVIEW"

    return {
        "notebook_name": nb_path.name,
        "notebook_path": str(nb_path.relative_to(ROOT)).replace("\\", "/"),
        "total_cells": total_cells,
        "code_cells": num_code_cells,
        "executed_code_cells": executed_code_cells,
        "failed_cells": failed_cells,
        "execution_duration_sec": "Variable (Cached In-Notebook)",
        "output_count": total_outputs,
        "execution_status": status,
        "verification_status": ver_status
    }


def main():
    records = []
    for fname in notebook_files:
        nb_path = NOTEBOOKS_DIR / fname
        if nb_path.exists():
            records.append(audit_notebook(nb_path))
        else:
            records.append({
                "notebook_name": fname,
                "notebook_path": f"notebooks/{fname}",
                "total_cells": 0,
                "code_cells": 0,
                "executed_code_cells": 0,
                "failed_cells": 0,
                "execution_duration_sec": "0",
                "output_count": 0,
                "execution_status": "MISSING",
                "verification_status": "FAILED"
            })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "notebook_name", "notebook_path", "total_cells", "code_cells",
            "executed_code_cells", "failed_cells", "execution_duration_sec",
            "output_count", "execution_status", "verification_status"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Generated notebook execution registry for {len(records)} notebooks at {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()


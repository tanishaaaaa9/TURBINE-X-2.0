"""TURBINE-X Notebook Batch Executor.
Executes all 19 notebooks sequentially, capturing full execution outputs, figures, and tracebacks.
"""
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"

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


def execute_notebook(nb_path: Path):
    print(f"\n>>> Executing: {nb_path.name}")
    t0 = time.time()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}})
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    elapsed = time.time() - t0
    print(f"    [COMPLETED] in {elapsed:.2f}s")


def main():
    print(f"Starting execution of {len(notebook_files)} notebooks in {NOTEBOOKS_DIR}...")
    success = 0
    failed = []

    for nb_name in notebook_files:
        nb_path = NOTEBOOKS_DIR / nb_name
        try:
            execute_notebook(nb_path)
            success += 1
        except Exception as e:
            print(f"    [FAILED] {nb_name}: {e}")
            failed.append((nb_name, str(e)))

    print(f"\nExecution Finished: {success}/{len(notebook_files)} succeeded.")
    if failed:
        print("Failures:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    else:
        print("ALL NOTEBOOKS EXECUTED WITH 100% SUCCESS!")


if __name__ == "__main__":
    main()


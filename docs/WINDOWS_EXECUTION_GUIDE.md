# TURBINE-X: Windows Execution Guide

## 1. Environment Details on Windows
- **OS:** Windows 11 64-bit
- **CPU:** AMD Ryzen 9 8940HX (16 Cores, 32 Threads)
- **RAM:** 16GB
- **Target Mode:** CPU-first execution (NVIDIA CUDA optional; all models operate natively on CPU)
- **Python Version:** Python 3.13.14 (fully compatible with Scikit-Learn 1.9, XGBoost 2.1, LightGBM 4.7, CatBoost 1.2, Optuna 5.0, FLAML 2.7, SHAP 0.54, Streamlit 1.66)

---

## 2. Windows Batch Execution Scripts
The repository root contains 4 automated Windows `.bat` scripts:

1. **`RUN_ON_WINDOWS.bat`**
   - Automatically detects Python interpreter from parent `.venv` (`..\.venv\Scripts\python.exe`), local `.venv`, or system PATH.
   - Executes the quick benchmark verification (`scripts\run_project.py --quick`).
   - Executes the complete test suite (`pytest tests -v`).

2. **`RUN_QUICK_TEST.bat`**
   - Runs fast model validation (7 models, reduced tree counts) across Track A and Track B.
   - Saves metrics to `outputs/tables/locked_test_metrics.csv` and serializes model artifacts to `artifacts/models/`.

3. **`RUN_FULL_BENCHMARK.bat`**
   - Trains the full ensemble of 12 model architectures across Track A and Track B.
   - Runs chronological within-development cross-validation and 2014 calibration.

4. **`RUN_DASHBOARD.bat`**
   - Launches the interactive Streamlit intelligence dashboard on `http://localhost:8501`.

---

## 3. PowerShell Commands Reference
For developers working directly in Windows PowerShell:

```powershell
# Set Execution Policy if needed
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate Virtual Environment
..\.venv\Scripts\Activate.ps1

# Run Quick Benchmark
python scripts\run_project.py --quick

# Run Full Model Benchmark
python scripts\run_project.py

# Run Automated Test Suite
python -m pytest tests -v

# Launch Streamlit Application
streamlit run app.py
```


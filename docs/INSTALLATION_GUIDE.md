# TURBINE-X: Installation & Environment Setup Guide

## 1. Prerequisites
- **Operating System:** Windows 10/11, Linux (Ubuntu 20.04+), or macOS (Sonoma+)
- **Python Version:** Python 3.11 to 3.13 (Python 3.13.14 verified)
- **Virtual Environment:** Recommended virtual environment tool (`venv` or `conda`)
- **Hardware Requirements:** CPU-first benchmarking. Minimum 8GB RAM (16GB recommended). No GPU required.

---

## 2. Step-by-Step Installation

### Step 2.1: Clone or Open the Repository
```powershell
cd D:\TURBINE_X_Full_Project\TURBINE_X_Full_Project
```

### Step 2.2: Virtual Environment Configuration
If a virtual environment is already present in the parent directory (`..\.venv`), it can be used directly. To create a fresh local environment:
```powershell
python -m venv .venv
.venv\Scripts\activate.bat
```

### Step 2.3: Upgrade Package Installer & Install Requirements
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2.4: Register Jupyter Kernel (Optional)
```powershell
python -m ipykernel install --user --name turbine-x --display-name "Python 3.13 (TURBINE-X)"
```

---

## 3. Verification of Installation
Run the automated verification suite:
```powershell
python scripts\run_project.py --quick
python -m pytest tests -v
```
If all 18 unit tests pass, the environment is 100% verified.


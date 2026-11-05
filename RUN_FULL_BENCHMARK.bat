@echo off
setlocal
cd /d "%~dp0"
if exist "TURBINE_X_Full_Project" (
    cd /d "%~dp0TURBINE_X_Full_Project"
)
echo ========================================================
echo   TURBINE-X: Running Comprehensive Full ML Benchmark
echo ========================================================

set "PYTHON_EXE="
if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

%PYTHON_EXE% scripts\run_project.py
pause

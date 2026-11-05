@echo off
setlocal
cd /d "%~dp0"
if exist "TURBINE_X_Full_Project" (
    cd /d "%~dp0TURBINE_X_Full_Project"
)
echo ========================================================
echo   Launching TURBINE-X 2.0 Master Execution Pipeline
echo ========================================================

set "PYTHON_EXE="
if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

%PYTHON_EXE% src\pipeline\run_turbine_x_2.py
pause

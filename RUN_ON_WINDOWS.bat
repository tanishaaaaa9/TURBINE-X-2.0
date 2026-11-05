@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
if exist "TURBINE_X_Full_Project" (
    cd /d "%~dp0TURBINE_X_Full_Project"
)
echo ========================================================
echo   TURBINE-X: Windows Environment & Execution Setup
echo ========================================================

rem Locate Python from parent .venv, local .venv, or PATH
set "PYTHON_EXE="
if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\.venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python not found in ..\.venv, .\.venv, or system PATH.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
%PYTHON_EXE% -V
echo.
echo Running Quick Project Verification...
%PYTHON_EXE% scripts\run_project.py --quick

echo.
echo Running Pytest Suite...
%PYTHON_EXE% -m pytest tests -v

echo.
echo ========================================================
echo TURBINE-X setup and quick verification completed!
echo To run the interactive dashboard: RUN_DASHBOARD.bat
echo To run full benchmark: RUN_FULL_BENCHMARK.bat
echo ========================================================
pause

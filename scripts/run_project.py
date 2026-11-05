"""Run the reproducible conventional-ML benchmark from the repository root."""
import os
import sys
import subprocess
from pathlib import Path

# Auto-detect and relaunch with the project virtual environment if running under global python
def _ensure_venv():
    # If CatBoost is already importable, we're in the right environment
    try:
        import catboost  # noqa: F401
        return
    except ImportError:
        pass

    # Look for the project's .venv python
    curr = Path(__file__).resolve().parent
    candidates = [
        curr.parent.parent / ".venv" / "Scripts" / "python.exe",
        curr.parent / ".venv" / "Scripts" / "python.exe",
        curr.parent.parent / ".venv" / "bin" / "python",
        curr.parent / ".venv" / "bin" / "python",
    ]
    for cand in candidates:
        if cand.exists():
            print(f"[TURBINE-X Launcher] Auto-bridging to project virtual environment: {cand}")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            cmd = [str(cand), str(Path(__file__).resolve())] + sys.argv[1:]
            res = subprocess.run(cmd, env=env, cwd=str(curr.parent))
            sys.exit(res.returncode)

_ensure_venv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline.run_experiments import run
import argparse

p = argparse.ArgumentParser(description="TURBINE-X ML Benchmark Runner")
p.add_argument("--quick", action="store_true", help="Run fast subset benchmark")
p.add_argument("--seed", type=int, default=42, help="Random seed")
a = p.parse_args()

run(quick=a.quick, seed=a.seed)

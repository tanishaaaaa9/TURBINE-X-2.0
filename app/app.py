"""TURBINE-X Streamlit App launcher shim."""
from pathlib import Path
import sys

# Ensure root is in path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

# Run main app script
app_file = root / "app.py"
if app_file.exists():
    with open(app_file, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, {"__name__": "__main__", "__file__": str(app_file)})


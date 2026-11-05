import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.tracker import generate_report
if __name__ == "__main__":
    summary = generate_report()
    print(f"Overall: {summary['overall_percent']:.1f}%")
    print(f"Current phase: {summary['current_phase']}")
    if summary["next_task"]:
        print("Next:", summary["next_task"]["task_id"], summary["next_task"]["title"])

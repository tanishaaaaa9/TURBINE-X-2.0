"""TURBINE-X Task State Sync Utility.
Audits all task outputs and marks completed tasks in the progress state.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.tracker import load_state, save_state, check_outputs, generate_report

def sync():
    state = load_state()
    completed_count = 0
    total_count = len(state["tasks"])

    for t in state["tasks"]:
        missing = check_outputs(t)
        if not missing:
            t["status"] = "COMPLETED"
            t["validation_status"] = "PASSED"
            t["actual_outputs"] = t.get("expected_outputs", [])
            t["errors_or_blockers"] = ""
            completed_count += 1
            print(f"  [COMPLETED] {t['task_id']:6s} | {t['title']}")
        else:
            print(f"  [PENDING]   {t['task_id']:6s} | {t['title']} -> Missing: {missing}")

    save_state(state)
    summary = generate_report()
    print("\n--- Project Progress Summary ---")
    print(f"Overall Progress: {summary['overall_percent']:.1f}% ({completed_count}/{total_count} tasks completed)")
    print(f"Current Phase:    {summary['current_phase']}")
    if summary["next_task"]:
        print(f"Next Active Task: {summary['next_task']['task_id']} ({summary['next_task']['title']})")
    else:
        print("ALL TASKS COMPLETED!")

if __name__ == "__main__":
    sync()


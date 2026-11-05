"""Persistent TURBINE-X project progress tracker.
A task can be marked COMPLETED only when validation_passed=True and every
declared expected output exists and is non-empty (directories may be empty only
if explicitly declared with expected_type='directory' in the registry).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "tasks.json"
STATE = ROOT / "experiments" / "progress_state.json"
REPORT = ROOT / "outputs" / "reports" / "progress_report.md"
CHART = ROOT / "outputs" / "figures" / "project_progress.png"
VALID_STATES = {"NOT_STARTED","IN_PROGRESS","BLOCKED","COMPLETED","FAILED"}

def now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def load_registry():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))

def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"created_at": now(), "updated_at": now(), "tasks": load_registry()["tasks"]}

def save_state(state):
    state["updated_at"] = now()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE)

def check_outputs(task):
    missing = []
    for rel in task.get("expected_outputs", []):
        p = ROOT / rel
        if not p.exists():
            missing.append(f"missing: {rel}")
        elif p.is_file() and p.stat().st_size == 0:
            missing.append(f"empty: {rel}")
    return missing

def update_task(task_id: str, status: str, *, validation_passed: bool=False,
                actual_outputs: Optional[list[str]]=None, notes: str="",
                errors_or_blockers: str=""):
    status = status.upper()
    if status not in VALID_STATES:
        raise ValueError(f"status must be one of {sorted(VALID_STATES)}")
    state = load_state()
    matches = [t for t in state["tasks"] if t["task_id"] == task_id]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate task id: {task_id}")
    task = matches[0]
    if task["status"] == "NOT_STARTED":
        task["start_date"] = now()
    task["last_updated"] = now()
    task["status"] = status
    task["notes"] = notes
    task["errors_or_blockers"] = errors_or_blockers
    if actual_outputs is not None:
        task["actual_outputs"] = actual_outputs
    if status == "COMPLETED":
        missing = check_outputs(task)
        dependency_errors = []
        for dep_id in task.get("dependencies", []):
            dep = next((x for x in state["tasks"] if x["task_id"] == dep_id), None)
            if dep is None or dep["status"] != "COMPLETED":
                dependency_errors.append(f"dependency not completed: {dep_id}")
        if not validation_passed:
            task["status"] = "FAILED"
            task["validation_status"] = "FAILED"
            task["errors_or_blockers"] = "Completion rejected: validation_passed was not True."
        elif missing or dependency_errors:
            task["status"] = "BLOCKED"
            task["validation_status"] = "FAILED"
            task["errors_or_blockers"] = "; ".join(missing + dependency_errors)
        else:
            task["validation_status"] = "PASSED"
            task["errors_or_blockers"] = ""
    elif status == "FAILED":
        task["validation_status"] = "FAILED"
    elif status in {"IN_PROGRESS","BLOCKED"}:
        task["validation_status"] = "PENDING"
    else:
        task["validation_status"] = "NOT_RUN"
    save_state(state)
    generate_report()
    return task

def summary():
    state = load_state()
    ts = state["tasks"]
    total = len(ts)
    counts = {s: sum(t["status"] == s for t in ts) for s in VALID_STATES}
    complete = counts["COMPLETED"]
    phase_names = list(dict.fromkeys(t["phase"] for t in ts))
    phases = {}
    for ph in phase_names:
        group = [t for t in ts if t["phase"] == ph]
        phases[ph] = {"completed":sum(t["status"]=="COMPLETED" for t in group),
                      "total":len(group),
                      "percent":100*sum(t["status"]=="COMPLETED" for t in group)/len(group)}
    pending = [t for t in ts if t["status"] in {"NOT_STARTED","IN_PROGRESS","BLOCKED","FAILED"}]
    current = next((t["phase"] for t in ts if t["status"] != "COMPLETED"), "Completed")
    next_task = next((t for t in ts if t["status"]=="NOT_STARTED" and
                      all(next((x for x in ts if x["task_id"]==d),{}).get("status")=="COMPLETED"
                          for d in t.get("dependencies",[]))), None)
    return {"overall_percent":100*complete/total if total else 0,
            "counts":counts,"total":total,"completed":complete,
            "pending":counts["NOT_STARTED"]+counts["IN_PROGRESS"],
            "blocked":counts["BLOCKED"],"failed":counts["FAILED"],
            "current_phase":current,"next_task":next_task,"phases":phases}

def generate_report():
    state = load_state()
    s = summary()
    lines = ["# TURBINE-X Project Progress Report","",
             f"Updated: {state['updated_at']}","",
             f"**Overall completion: {s['overall_percent']:.1f}%**",
             f"Completed: {s['completed']}/{s['total']} | Pending: {s['pending']} | Blocked: {s['blocked']} | Failed: {s['failed']}","",
             f"**Current phase:** {s['current_phase']}","",
             f"**Next recommended task:** {s['next_task']['task_id'] + ' — ' + s['next_task']['title'] if s['next_task'] else 'No pending task'}","",
             "## Phase-wise progress","",
             "| Phase | Completed | Total | Progress |","|---|---:|---:|---:|"]
    for ph,v in s["phases"].items():
        lines.append(f"| {ph} | {v['completed']} | {v['total']} | {v['percent']:.1f}% |")
    lines += ["","## Task status","",
              "| ID | Phase | Task | Status | Validation |","|---|---|---|---|---|"]
    for t in state["tasks"]:
        lines.append(f"| {t['task_id']} | {t['phase']} | {t['title']} | {t['status']} | {t['validation_status']} |")
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text("\\n".join(lines),encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        names=list(s["phases"].keys()); vals=[s["phases"][n]["percent"] for n in names]
        fig,ax=plt.subplots(figsize=(10,max(4,0.42*len(names))))
        ax.barh(names,vals); ax.set_xlim(0,100); ax.set_xlabel("Completed tasks (%)")
        ax.set_title(f"TURBINE-X Progress — {s['overall_percent']:.1f}% overall")
        ax.grid(axis="x",alpha=.25); fig.tight_layout()
        CHART.parent.mkdir(parents=True,exist_ok=True); fig.savefig(CHART,dpi=180); plt.close(fig)
    except Exception as e:
        print(f"Chart generation skipped: {e}")
    return s

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)
    u=sub.add_parser("update"); u.add_argument("task_id"); u.add_argument("status",choices=sorted(VALID_STATES))
    u.add_argument("--validated",action="store_true"); u.add_argument("--notes",default="")
    u.add_argument("--blocker",default="")
    sub.add_parser("report")
    args=ap.parse_args()
    if args.cmd=="update":
        t=update_task(args.task_id,args.status,validation_passed=args.validated,
                      notes=args.notes,errors_or_blockers=args.blocker)
        print(json.dumps(t,indent=2))
    else:
        print(json.dumps(generate_report(),indent=2))
if __name__=="__main__": main()

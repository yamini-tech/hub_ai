import json
import os
from datetime import datetime
from typing import Any


def aggregate_results(
    case_results: list[dict[str, Any]],
    run_config: dict[str, Any],
    results_dir: str = "evals/results",
) -> dict[str, Any]:
    total = len(case_results)
    passed = sum(1 for c in case_results if c.get("passed", False))
    failed = total - passed

    by_task: dict[str, dict[str, Any]] = {}
    for c in case_results:
        task = c.get("task", "unknown")
        if task not in by_task:
            by_task[task] = {"total": 0, "passed": 0, "failed": 0, "scores": []}
        by_task[task]["total"] += 1
        if c.get("passed"):
            by_task[task]["passed"] += 1
        else:
            by_task[task]["failed"] += 1
        if c.get("judge_score") is not None:
            by_task[task]["scores"].append(c["judge_score"])

    task_summaries = {}
    for task, data in sorted(by_task.items()):
        avg = round(sum(data["scores"]) / len(data["scores"]), 2) if data["scores"] else None
        task_summaries[task] = {
            "total": data["total"],
            "passed": data["passed"],
            "failed": data["failed"],
            "avg_judge_score": avg,
        }

    all_scores = [c["judge_score"] for c in case_results if c.get("judge_score") is not None]
    overall_avg = round(sum(all_scores) / len(all_scores), 2) if all_scores else None

    report = {
        "run_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "config": run_config,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": round(passed / total * 100, 1) if total else 0,
            "avg_judge_score": overall_avg,
            "by_task": task_summaries,
        },
        "cases": case_results,
    }

    os.makedirs(results_dir, exist_ok=True)
    filename = f"{report['run_id']}.json"
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    link_path = os.path.join(results_dir, "latest.json")
    if os.path.lexists(link_path):
        os.remove(link_path)
    os.symlink(filename, link_path)

    return report


def print_summary(report: dict[str, Any]) -> None:
    s = report["summary"]
    sep = "=" * 58
    print(f"\n{sep}")
    print(f"  Eval Results — {report['run_id']}")
    print(f"{sep}")
    print(
        f"  Total:  {s['total']:3d}   Passed: {s['passed']:3d}   "
        f"Failed: {s['failed']:3d}   Rate: {s['pass_rate_pct']}%"
    )
    if s["avg_judge_score"] is not None:
        print(f"  Avg Judge Score: {s['avg_judge_score']:.2f} / 5.00")
    print(f"{sep}")
    print("  By Task:")
    for task_name, ts in sorted(s["by_task"].items()):
        score_str = f"  Judge: {ts['avg_judge_score']:.2f}" if ts["avg_judge_score"] is not None else ""
        print(f"    {task_name:12s}  {ts['passed']:2d}/{ts['total']:2d} passed  {score_str}")
    print(f"{sep}")
    print(f"  Report written to: {report.get('_filepath', '')}")
    print(f"{sep}\n")


def print_case_result(case_id: str, passed: bool, detail: str, judge_score=None) -> None:
    status = "PASS" if passed else "FAIL"
    score = f"  [{judge_score}/5]" if judge_score is not None else ""
    print(f"    {status:4s}  {case_id:15s}{score}  {detail}")

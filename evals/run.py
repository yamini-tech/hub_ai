#!/usr/bin/env python3
"""Eval harness for SmartHub AI Brain.

Usage:
  python evals/run.py                                        # all tasks, objective checks only
  python evals/run.py --tasks summarize,parse                # specific tasks
  python evals/run.py --judge-model openai/gpt-4o            # with LLM-as-judge
  python evals/run.py --base http://localhost:8003           # custom base URL
  python evals/run.py --threshold 4.0                        # custom pass threshold
"""

import argparse
import json
import os
import sys
import asyncio
import httpx
from pathlib import Path

from evals.judge import judge_response
from evals.reporter import aggregate_results, print_summary, print_case_result

EVALS_DIR = Path(__file__).parent
DATASETS_DIR = EVALS_DIR / "datasets"
RESULTS_DIR = EVALS_DIR / "results"
DEFAULT_BASE = "http://localhost:8003"
DEFAULT_THRESHOLD = 0.0
HEADERS = {"Content-Type": "application/json"}
TIMEOUT = 60.0


def load_datasets(tasks: list[str] | None = None) -> list[dict]:
    datasets = []
    for fpath in sorted(DATASETS_DIR.glob("*.json")):
        with open(fpath) as f:
            ds = json.load(f)
        if tasks is None or ds["task"] in tasks:
            datasets.append(ds)
    return datasets


def _deep_get(obj, dotted_key: str):
    parts = dotted_key.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.lstrip("-").isdigit():
            idx = int(part)
            current = current[idx] if 0 <= idx < len(current) else None
        else:
            return None
        if current is None:
            return None
    return current


async def run_objective_checks(response_text: str, expected: dict, response_json: dict | None = None) -> tuple[bool, list[str]]:
    failures = []

    if expected.get("min_length_chars"):
        if len(response_text) < expected["min_length_chars"]:
            failures.append(
                f"too short ({len(response_text)} < {expected['min_length_chars']} chars)"
            )

    for term in expected.get("must_contain", []):
        if term.lower() not in response_text.lower():
            failures.append(f"missing expected term: '{term}'")

    for term in expected.get("must_not_contain", []):
        if term.lower() in response_text.lower():
            failures.append(f"contains forbidden term: '{term}'")

    if expected.get("must_contain_keys") and response_json is not None:
        for key in expected["must_contain_keys"]:
            if _deep_get(response_json, key) is None:
                failures.append(f"missing expected key in parsed output: '{key}'")

    return len(failures) == 0, failures


async def run_case(
    client: httpx.AsyncClient,
    dataset: dict,
    case: dict,
    judge_model: str | None,
    base_url: str = DEFAULT_BASE,
) -> dict:
    task = dataset["task"]
    case_id = case["id"]
    input_text = case["input"]
    expected = case.get("expected", {})
    endpoint = dataset["endpoint"]
    response_key = dataset["response_key"]
    url = f"{base_url}{endpoint}"

    body = {"text": input_text, "session_id": f"eval-{case_id}"}

    if dataset["task"] == "parse" and case.get("json_schema"):
        body["json_schema"] = case["json_schema"]

    try:
        r = await client.post(url, json=body, timeout=TIMEOUT)
    except Exception as e:
        return {
            "id": case_id,
            "task": task,
            "passed": False,
            "error": f"Request failed: {e}",
            "judge_score": None,
            "judge_reasoning": None,
            "response": None,
        }

    expected_status = expected.get("expected_http_status", 200)

    if r.status_code != expected_status:
        return {
            "id": case_id,
            "task": task,
            "passed": False,
            "error": f"HTTP {r.status_code} (expected {expected_status})",
            "judge_score": None,
            "judge_reasoning": None,
            "response": r.text[:500],
        }

    if expected_status != 200:
        return {
            "id": case_id,
            "task": task,
            "passed": True,
            "error": None,
            "judge_score": None,
            "judge_reasoning": None,
            "response": r.text[:200],
        }

    try:
        data = r.json()
    except Exception as e:
        return {
            "id": case_id,
            "task": task,
            "passed": False,
            "error": f"Invalid JSON response: {e}",
            "judge_score": None,
            "judge_reasoning": None,
            "response": r.text[:500],
        }

    response_text = data.get(response_key, "")
    if not response_text:
        return {
            "id": case_id,
            "task": task,
            "passed": False,
            "error": f"Response missing key '{response_key}'",
            "judge_score": None,
            "judge_reasoning": None,
            "response": json.dumps(data)[:500],
        }

    response_json = data.get(response_key) if isinstance(data.get(response_key), (dict, list)) else None
    response_str = str(response_text) if not isinstance(response_text, str) else response_text

    passed, failures = await run_objective_checks(response_str, expected, response_json)
    detail = "; ".join(failures) if failures else "ok"

    judge_score = None
    judge_reasoning = None
    if judge_model and passed:
        score, reasoning = await judge_response(
            task=task,
            user_input=input_text,
            response=response_str,
            criteria=expected.get("judge_criteria", ""),
            model=judge_model,
        )
        judge_score = score
        judge_reasoning = reasoning

    return {
        "id": case_id,
        "task": task,
        "passed": passed,
        "error": detail if not passed else None,
        "judge_score": judge_score,
        "judge_reasoning": judge_reasoning,
        "response": response_str[:300],
    }


async def main_async(args: argparse.Namespace) -> int:
    datasets = load_datasets(args.tasks)
    if not datasets:
        print("No datasets found or no tasks matched.")
        return 1

    base = args.base or DEFAULT_BASE
    total_cases = sum(len(ds["cases"]) for ds in datasets)
    print(f"\nEval Run — {len(datasets)} datasets, {total_cases} cases")
    print(f"Target: {base}")
    if args.judge_model:
        print(f"Judge:  {args.judge_model}")
    else:
        print(f"Judge:  disabled (objective checks only)")
    print(f"{'=' * 58}")

    case_results = []
    async with httpx.AsyncClient(base_url=base, headers=HEADERS) as client:
        for ds in datasets:
            print(f"\n  Task: {ds['task']} ({ds.get('description', '')})")
            for case in ds["cases"]:
                result = await run_case(client, ds, case, args.judge_model, base_url=base)
                case_results.append(result)
                score_str = f"  [{result['judge_score']}/5]" if result["judge_score"] is not None else ""
                status = "PASS" if result["passed"] else "FAIL"
                err = f"  {result['error']}" if result.get("error") else ""
                print(f"    {status:4s}  {case['id']:15s}{score_str}{err}")

    run_config = {
        "base_url": base,
        "judge_model": args.judge_model,
        "threshold": args.threshold,
    }
    report = aggregate_results(case_results, run_config, results_dir=str(RESULTS_DIR))
    print_summary(report)

    if report["summary"]["passed"] < report["summary"]["total"]:
        return 1
    if args.threshold and report["summary"].get("avg_judge_score") is not None:
        if report["summary"]["avg_judge_score"] < args.threshold:
            print(f"Average judge score {report['summary']['avg_judge_score']:.2f} below threshold {args.threshold}")
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SmartHub AI Brain Eval Harness")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base URL of the hub_ai service")
    parser.add_argument("--tasks", type=lambda s: [t.strip() for t in s.split(",")], help="Comma-separated task types to eval")
    parser.add_argument("--judge-model", default=None, help="Model to use as LLM judge (e.g. openai/gpt-4o)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Minimum average judge score (default: no threshold)")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

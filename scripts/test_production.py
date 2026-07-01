#!/usr/bin/env python3
"""Production-style test harness for SmartHub AI Brain.

Usage:
  SMARTHUB_TEST_MODEL=ollama/llama3.2 python scripts/test_production.py

Tests like a real client: SSE parsing, sync responses, async jobs, auth, health.
"""

import json
import sys
import time

import httpx

BASE = "http://localhost:8003"
HEADERS = {"Content-Type": "application/json"}
TIMEOUT = 30

passed = 0
failed = 0


def test(name):
    def decorator(fn):
        def wrapper(*a, **kw):
            global passed, failed
            try:
                fn(*a, **kw)
                print(f"  PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1

        return wrapper

    return decorator


@test("GET /health returns 200")
def test_health():
    r = httpx.get(f"{BASE}/health", timeout=TIMEOUT)
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


@test("GET /ready returns 200")
def test_ready():
    r = httpx.get(f"{BASE}/ready", timeout=TIMEOUT)
    assert r.status_code == 200
    assert "status" in r.json()


@test("GET / returns service info")
def test_root():
    r = httpx.get(f"{BASE}/", timeout=TIMEOUT)
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "SmartHub AI Brain"
    assert "endpoints" in data


@test("POST /api/ai/summarize/sync returns summary")
def test_summarize_sync():
    r = httpx.post(
        f"{BASE}/api/ai/summarize/sync",
        headers=HEADERS,
        json={"text": "Artificial intelligence is transforming business operations.", "session_id": "prod-sum"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data
    assert isinstance(data["summary"], str)
    assert len(data["summary"]) > 0


@test("POST /api/ai/summarize streams JSON tokens")
def test_summarize_stream():
    full = ""
    with httpx.Client(timeout=TIMEOUT) as client:
        with client.stream(
            "POST",
            f"{BASE}/api/ai/summarize",
            headers=HEADERS,
            json={"text": "Machine learning enables systems to learn from experience.", "session_id": "prod-stream"},
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            for line in r.iter_lines():
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
                    if "token" in payload:
                        full += payload["token"]
    assert len(full) > 0, "Stream returned no content"


@test("POST /api/ai/parse/sync returns parsed JSON")
def test_parse_sync():
    r = httpx.post(
        f"{BASE}/api/ai/parse/sync",
        headers=HEADERS,
        json={"text": "My name is John Doe. I live in New York.", "session_id": "prod-parse"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    data = r.json()
    assert "parsed" in data
    assert len(data["parsed"]) > 0


@test("POST /api/agent sync returns answer")
def test_agent_sync():
    r = httpx.post(
        f"{BASE}/api/agent/sync",
        headers=HEADERS,
        json={"text": "What is machine learning?", "session_id": "prod-agent"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data


@test("POST /api/agent streams JSON tokens")
def test_agent_stream():
    full = ""
    with httpx.Client(timeout=TIMEOUT) as client:
        with client.stream(
            "POST",
            f"{BASE}/api/agent",
            headers=HEADERS,
            json={"text": "Hello", "session_id": "prod-agent-stream"},
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            for line in r.iter_lines():
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if line.startswith("data: __tool_calls__:"):
                    continue
                if line.startswith("data: "):
                    payload = json.loads(line[6:])
                    if "token" in payload:
                        full += payload["token"]
    assert len(full) > 0, "Agent stream returned no content"


@test("POST /api/ai/process creates job + polling completes")
def test_process():
    r = httpx.post(
        f"{BASE}/api/ai/process",
        headers=HEADERS,
        json={
            "text": "Summarize this: AI is transforming the world.",
            "task_type": "summarize",
            "session_id": "prod-process",
        },
        timeout=TIMEOUT,
    )
    assert r.status_code == 202
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "processing"
    job_id = data["job_id"]

    for _ in range(10):
        r = httpx.get(f"{BASE}/api/ai/status/{job_id}", timeout=TIMEOUT)
        assert r.status_code == 200
        status_data = r.json()
        if status_data.get("status") == "completed":
            return
        time.sleep(1)
    raise AssertionError(f"Job {job_id} did not complete within 10s")


@test("GET /metrics returns Prometheus data")
def test_metrics():
    r = httpx.get(f"{BASE}/metrics", timeout=TIMEOUT)
    assert r.status_code == 200
    assert "http_requests_total" in r.text
    assert "llm_requests_total" in r.text


@test("Token limit exceeded returns 429")
def test_token_limit():
    long_text = "x " * 25000
    r = httpx.post(
        f"{BASE}/api/ai/summarize/sync",
        headers=HEADERS,
        json={"text": long_text, "session_id": "prod-limit"},
        timeout=TIMEOUT,
    )
    assert r.status_code == 429


@test("Non-existent job returns 404")
def test_job_not_found():
    r = httpx.get(f"{BASE}/api/ai/status/nonexistent-job-id", timeout=TIMEOUT)
    assert r.status_code == 404


def main():
    print("\nSmartHub AI Brain — Production Test Suite")
    print(f"Target: {BASE}")
    print(f"{'='*50}")

    tests = [
        test_health,
        test_ready,
        test_root,
        test_summarize_sync,
        test_summarize_stream,
        test_parse_sync,
        test_agent_sync,
        test_agent_stream,
        test_process,
        test_metrics,
        test_token_limit,
        test_job_not_found,
    ]

    for t in tests:
        t()

    print(f"{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

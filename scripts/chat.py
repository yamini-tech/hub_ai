#!/usr/bin/env python3
"""Interactive terminal chat with the SmartHub AI Brain.

Usage:
    python scripts/chat.py
"""

import httpx

BASE = "http://localhost:8003"
SESSION_ID = "cli-chat"
TIMEOUT = 120


def chat():
    print("SmartHub AI Brain — Interactive Chat")
    print("Type 'exit' or 'quit' to end.\n")

    with httpx.Client(timeout=TIMEOUT) as client:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            try:
                r = client.post(
                    f"{BASE}/api/agent/sync",
                    json={
                        "text": user_input,
                        "session_id": SESSION_ID,
                    },
                )
                r.raise_for_status()
                data = r.json()
                answer = data.get("answer", "")
                print(f"\nAgent: {answer}\n")
            except httpx.ConnectError:
                print("\nError: Cannot connect to server. Is it running on port 8003?\n")
            except httpx.HTTPStatusError as e:
                print(f"\nError: {e.response.status_code} — {e.response.text}\n")
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    chat()

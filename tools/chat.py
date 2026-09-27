#!/usr/bin/env python3
"""
Interactive Multi-Turn CLI Chat Client for colab-model-station.
Supports streaming responses from vLLM, Ollama, or OpenAI-compatible endpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def chat_loop(endpoint: str, model: str) -> None:
    api_url = f"{endpoint.rstrip('/')}/v1/chat/completions"

    print("=" * 70)
    print("Colab Model Station - Interactive LLM Shell")
    print(f"Endpoint: {api_url} | Model: [{model}]")
    print("Commands: Type 'exit' to quit, 'clear' to reset context")
    print("=" * 70)

    messages = [
        {"role": "system", "content": "You are an expert AI software engineer. Provide concise, clear, and accurate answers."}
    ]

    while True:
        try:
            user_input = input("\n[User] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Exiting session.")
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": "You are an expert AI software engineer. Provide concise, clear, and accurate answers."}]
            print("[System] Conversation context reset.")
            continue

        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
        }

        req = urllib.request.Request(
            api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer token",
            },
        )

        print(f"\n[{model}] > ", end="", flush=True)
        assistant_reply = ""
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            print(content, end="", flush=True)
                            assistant_reply += content
                    except json.JSONDecodeError:
                        pass
            print()
            messages.append({"role": "assistant", "content": assistant_reply})
        except Exception as e:
            print(f"\n[ERROR] Request failed: {e}")
            print(f"Verify that the LLM engine is running at {endpoint}.")


def main():
    parser = argparse.ArgumentParser(description="Colab Model Station LLM Chat Interface")
    parser.add_argument("--endpoint", default=os.environ.get("STATION_LLM_ENDPOINT", "http://127.0.0.1:8000"))
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    args = parser.parse_args()

    chat_loop(args.endpoint, args.model)


if __name__ == "__main__":
    main()

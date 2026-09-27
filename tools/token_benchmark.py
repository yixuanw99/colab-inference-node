#!/usr/bin/env python3
"""
Performance & Token Throughput Benchmark Suite for colab-model-station LLMs.
Measures TTFT (Time-To-First-Token), tokens per second, and latency over OpenAI /v1 endpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request


def run_token_benchmark(endpoint: str, model: str, prompt: str) -> None:
    api_url = f"{endpoint.rstrip('/')}/v1/chat/completions"

    print("=" * 70)
    print("Colab Model Station - LLM Token Benchmark")
    print(f"Endpoint: {api_url} | Model: {model}")
    print(f"Prompt:   '{prompt}'")
    print("=" * 70)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0.2,
    }

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer token",
        },
    )

    t0 = time.time()
    first_token_time = None
    full_text = ""

    print("\n[Streaming Response Begin]\n")
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
                    if not first_token_time:
                        first_token_time = time.time()
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "") or delta.get("reasoning", "") or delta.get("reasoning_content", "")
                        print(content, end="", flush=True)
                        full_text += content
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}", file=sys.stderr)
        return

    t1 = time.time()
    print("\n\n[Streaming Response End]\n")

    total_time = t1 - t0
    gen_time = (t1 - first_token_time) if first_token_time else total_time
    output_chars = len(full_text)
    approx_tokens = int(output_chars * 0.75) if any(ord(c) > 127 for c in full_text) else int(output_chars / 4)

    print("=" * 70)
    print("LLM Benchmark Metrics:")
    if first_token_time:
        print(f"  Time To First Token (TTFT): {first_token_time - t0:.3f} s")
    print(f"  Total Duration:             {total_time:.2f} s")
    print(f"  Generation Phase:           {gen_time:.2f} s")
    print(f"  Output Volume:              {output_chars} characters (~{approx_tokens} tokens)")
    if gen_time > 0 and approx_tokens > 0:
        print(f"  Estimated Throughput:       ~{approx_tokens / gen_time:.2f} tokens/s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="LLM Token Benchmark")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    parser.add_argument("--prompt", default="Explain the difference between PagedAttention and standard multi-head attention in 3 concise paragraphs.")
    args = parser.parse_args()

    run_token_benchmark(args.endpoint, args.model, args.prompt)


if __name__ == "__main__":
    main()

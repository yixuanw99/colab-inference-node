#!/usr/bin/env python3
"""
Performance & Latency Benchmarking Utility for colab-model-station.
Measures TTFT, generation latency, memory overhead, and throughput.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request


def run_benchmark(endpoint: str, prompt: str, steps: int, iterations: int, width: int, height: int) -> None:
    print("================================================================================")
    print("Colab Model Station - Performance Benchmark")
    print(f"Target Endpoint: {endpoint}")
    print(f"Resolution: {width}x{height} | Steps: {steps} | Iterations: {iterations}")
    print("================================================================================")

    # 1. Health check
    health_url = f"{endpoint.rstrip('/')}/health"
    try:
        t0 = time.time()
        with urllib.request.urlopen(health_url, timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        health_lat = (time.time() - t0) * 1000
        print(f"[HEALTH] Latency: {health_lat:.1f}ms | GPU: {health.get('gpu_name')} | VRAM Allocated: {health.get('vram_allocated_mb')}MB")
    except Exception as e:
        print(f"[ERROR] Endpoint unreachable: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Warmup run
    print("\n[WARMUP] Executing single warmup generation...")
    gen_url = f"{endpoint.rstrip('/')}/v1/images/generations"
    payload = json.dumps({
        "prompt": prompt,
        "num_inference_steps": steps,
        "width": width,
        "height": height,
        "return_base64": False,
    }).encode("utf-8")

    req = urllib.request.Request(gen_url, data=payload, headers={"Content-Type": "application/json"})
    t_start = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        warmup_res = json.loads(resp.read().decode("utf-8"))
    warmup_time = time.time() - t_start
    print(f"[WARMUP] Completed in {warmup_time:.2f}s (Engine: {warmup_res.get('generation_time_sec')}s)")

    # 3. Iterative benchmark
    print(f"\n[BENCHMARK] Running {iterations} consecutive iterations...")
    latencies: list[float] = []
    engine_times: list[float] = []

    for i in range(1, iterations + 1):
        t0 = time.time()
        req = urllib.request.Request(gen_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        elapsed = time.time() - t0
        latencies.append(elapsed)
        engine_times.append(res.get("generation_time_sec", elapsed))
        print(f"  Iteration {i}/{iterations}: Total: {elapsed:.2f}s | Compute: {res.get('generation_time_sec'):.2f}s")

    # 4. Summary metrics
    avg_total = statistics.mean(latencies)
    avg_compute = statistics.mean(engine_times)
    std_dev = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    steps_per_sec = steps / avg_compute if avg_compute > 0 else 0

    print("\n================================================================================")
    print("Benchmark Results Summary:")
    print(f"  Iterations:          {iterations}")
    print(f"  Mean Total Latency:  {avg_total:.2f}s (±{std_dev:.2f}s)")
    print(f"  Mean Compute Time:   {avg_compute:.2f}s")
    print(f"  Throughput:          {steps_per_sec:.2f} steps/second")
    print("================================================================================")


def main():
    parser = argparse.ArgumentParser(description="Inference Benchmark")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="A high quality cinematic photograph of an astronomical observatory at night")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    args = parser.parse_args()

    run_benchmark(args.endpoint, args.prompt, args.steps, args.iterations, args.width, args.height)


if __name__ == "__main__":
    main()

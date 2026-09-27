"""
LLM Service Runner for colab-model-station.
Supervises headless vLLM and Ollama daemon lifecycles.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "logs"
VLLM_PID_FILE = LOG_DIR / "vllm.pid"
VLLM_LOG_FILE = LOG_DIR / "vllm.log"
OLLAMA_PID_FILE = LOG_DIR / "ollama.pid"
OLLAMA_LOG_FILE = LOG_DIR / "ollama.log"


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ==============================================================================
# vLLM Engine Supervisor (OpenAI-compatible Port 8000)
# ==============================================================================
def start_vllm(
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
    port: int = 8000,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
    quantization: str = "auto",
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if VLLM_PID_FILE.is_file():
        try:
            pid = int(VLLM_PID_FILE.read_text().strip())
            if is_running(pid):
                print(f"[VLLM] Service already running (PID: {pid}).")
                return
        except Exception:
            pass

    cmd = [
        "vllm", "serve", model,
        "--port", str(port),
        "--trust-remote-code",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    if quantization != "auto" and quantization != "none":
        cmd.extend(["--quantization", quantization])

    print(f"[VLLM] Launching vLLM daemon: {' '.join(cmd)}")
    with open(VLLM_LOG_FILE, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    VLLM_PID_FILE.write_text(str(proc.pid))
    print(f"[VLLM] Process started (PID: {proc.pid}). Waiting for endpoint readiness...")

    for i in range(1, 180):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[SUCCESS] vLLM endpoint ready at http://127.0.0.1:{port}/v1")
                    return
        except Exception:
            pass

        if not is_running(proc.pid):
            print("[ERROR] vLLM process terminated unexpectedly during startup. Last logs:", file=sys.stderr)
            if VLLM_LOG_FILE.is_file():
                lines = VLLM_LOG_FILE.read_text().splitlines()[-25:]
                print("\n".join(lines), file=sys.stderr)
            sys.exit(1)

        if i % 15 == 0:
            print(f"[VLLM] Still compiling CUDA graphs and loading weights ({i * 2}s elapsed)...")
        time.sleep(2)

    print(f"[WARN] vLLM still loading after 6 minutes. Monitor with 'tail -f {VLLM_LOG_FILE}'.")


def stop_vllm() -> None:
    print("[VLLM] Terminating vLLM process...")
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    subprocess.run(["pkill", "-f", "VLLM::EngineCore"], check=False)
    if VLLM_PID_FILE.is_file():
        VLLM_PID_FILE.unlink()
    print("[SUCCESS] vLLM stopped.")


def status_vllm(port: int = 8000) -> None:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                print(f"[RUNNING] vLLM is active on port {port}.")
                return
    except Exception:
        pass
    print("[STOPPED] vLLM is not reachable.")


# ==============================================================================
# Ollama Engine Supervisor (GGUF Port 11434)
# ==============================================================================
def start_ollama(port: int = 11434) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/version")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                print(f"[OLLAMA] Service already active on port {port}.")
                return
    except Exception:
        pass

    env = os.environ.copy()
    env["OLLAMA_ORIGINS"] = "*"
    env["OLLAMA_HOST"] = f"0.0.0.0:{port}"

    print(f"[OLLAMA] Starting Ollama daemon on port {port}...")
    with open(OLLAMA_LOG_FILE, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(["ollama", "serve"], stdout=out, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)

    OLLAMA_PID_FILE.write_text(str(proc.pid))
    time.sleep(3)

    if is_running(proc.pid):
        print(f"[SUCCESS] Ollama running at http://127.0.0.1:{port}")
    else:
        print("[ERROR] Ollama failed to start. Last log lines:", file=sys.stderr)
        if OLLAMA_LOG_FILE.is_file():
            print("\n".join(OLLAMA_LOG_FILE.read_text().splitlines()[-15:]), file=sys.stderr)
        sys.exit(1)


def stop_ollama() -> None:
    print("[OLLAMA] Terminating Ollama daemon...")
    subprocess.run(["pkill", "-f", "ollama serve"], check=False)
    if OLLAMA_PID_FILE.is_file():
        OLLAMA_PID_FILE.unlink()
    print("[SUCCESS] Ollama stopped.")


def status_ollama(port: int = 11434) -> None:
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/version")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                print(f"[RUNNING] Ollama is active on port {port}.")
                return
    except Exception:
        pass
    print("[STOPPED] Ollama is not reachable.")


def pull_ollama(model: str) -> None:
    print(f"[OLLAMA] Pulling model: {model}...")
    res = subprocess.run(["ollama", "pull", model])
    if res.returncode != 0:
        sys.exit(res.returncode)


def list_ollama() -> None:
    subprocess.run(["ollama", "list"])


def main():
    parser = argparse.ArgumentParser(description="LLM Engine Supervisor")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # vllm
    vllm_p = subparsers.add_parser("vllm")
    vllm_sub = vllm_p.add_subparsers(dest="action", required=True)
    vllm_start = vllm_sub.add_parser("start")
    vllm_start.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    vllm_start.add_argument("--port", type=int, default=8000)
    vllm_start.add_argument("--max-len", type=int, default=8192)
    vllm_start.add_argument("--gpu-util", type=float, default=0.90)
    vllm_start.add_argument("--quantization", default="auto")

    vllm_sub.add_parser("stop")
    vllm_sub.add_parser("status")

    # ollama
    ollama_p = subparsers.add_parser("ollama")
    ollama_sub = ollama_p.add_subparsers(dest="action", required=True)
    ollama_start = ollama_sub.add_parser("start")
    ollama_start.add_argument("--port", type=int, default=11434)
    ollama_sub.add_parser("stop")
    ollama_sub.add_parser("status")
    ollama_pull = ollama_sub.add_parser("pull")
    ollama_pull.add_argument("model")
    ollama_sub.add_parser("list")

    args = parser.parse_args()
    if args.subcommand == "vllm":
        if args.action == "start":
            start_vllm(
                model=args.model,
                port=args.port,
                max_model_len=args.max_len,
                gpu_memory_utilization=args.gpu_util,
                quantization=args.quantization,
            )
        elif args.action == "stop":
            stop_vllm()
        elif args.action == "status":
            status_vllm()
    elif args.subcommand == "ollama":
        if args.action == "start":
            start_ollama(port=args.port)
        elif args.action == "stop":
            stop_ollama()
        elif args.action == "status":
            status_ollama()
        elif args.action == "pull":
            pull_ollama(args.model)
        elif args.action == "list":
            list_ollama()


if __name__ == "__main__":
    main()

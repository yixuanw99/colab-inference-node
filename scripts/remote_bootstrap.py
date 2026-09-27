#!/usr/bin/env python3
"""
Colab Model Station - Remote Headless Bootstrap Script
Executed on the remote Google Colab VM by `colab exec`.
Provisions dependencies, connects Tailscale mesh, launches target engine,
and arms the compute unit protection watchdog.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time


def log(msg: str) -> None:
    print(f"[REMOTE] {msg}", flush=True)


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    log(f"> {cmd}")
    res = subprocess.run(cmd, shell=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")
    return res


def main() -> None:
    repo_url = os.environ.get("STATION_REPO_URL", "https://github.com/yixuanw99/colab-inference-node.git")
    branch = os.environ.get("STATION_BRANCH", "main")
    engine = os.environ.get("STATION_ENGINE", "diffusers").lower()
    model = os.environ.get("STATION_MODEL", "black-forest-labs/FLUX.1-schnell")
    precision = os.environ.get("STATION_PRECISION", "auto")
    authkey = os.environ.get("TAILSCALE_AUTHKEY", "")
    ts_hostname = os.environ.get("TAILSCALE_HOSTNAME", "colab-model-station")
    hf_token = os.environ.get("HF_TOKEN", "")
    idle_timeout = os.environ.get("IDLE_TIMEOUT_SECONDS", "1800")

    station_dir = "/content/colab-inference-node"

    log("=" * 70)
    log("Colab Model Station - Remote Provisioning Sequence Initialized")
    log(f"Engine: {engine} | Model: {model} | Precision: {precision}")
    log("=" * 70)

    # 1. Hugging Face credentials
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        log("Configured Hugging Face token.")

    # 2. Clone or update repository
    if not os.path.exists(station_dir):
        log(f"Cloning {repo_url} (branch: {branch}) into {station_dir}...")
        run(f"git clone -b {branch} {repo_url} {station_dir}")
    else:
        log(f"Repository directory exists at {station_dir}; updating...")
        run(f"cd {station_dir} && git fetch && git checkout {branch} && git pull || true")

    # 3. Environment bootstrap
    log(f"Executing dependency bootstrap for '{engine}'...")
    run(f"cd {station_dir} && bash scripts/setup.sh {engine}")

    # 4. Tailscale Mesh Network Setup
    if authkey:
        log("Authenticating to Tailscale Mesh Network (Userspace Mode)...")
        run(f"cd {station_dir} && TAILSCALE_HOSTNAME='{ts_hostname}' bash engine.sh tunnel tailscale up '{authkey}'")
    else:
        log("Notice: TAILSCALE_AUTHKEY not provided; launching Tailscale daemon in unauthenticated mode.")
        run(f"cd {station_dir} && bash engine.sh tunnel tailscale up")

    # 5. Start Inference Engine
    log(f"Starting {engine.upper()} daemon...")
    if engine == "diffusers":
        start_cmd = f"cd {station_dir} && bash engine.sh diffusers start --model '{model}' --precision '{precision}'"
        run(start_cmd)
        run(f"cd {station_dir} && tailscale serve --bg --tcp 8000 8000 || true")
    elif engine == "comfyui":
        run(f"cd {station_dir} && bash engine.sh comfyui start")
        run(f"cd {station_dir} && tailscale serve --bg --tcp 8188 8188 || true")

    # 6. Start Compute Unit Watchdog
    log(f"Starting Idle Watchdog (Timeout: {idle_timeout}s)...")
    port = 8000 if engine == "diffusers" else 8188
    run(f"cd {station_dir} && bash engine.sh watchdog start --port {port} --timeout {idle_timeout}")

    # 7. Print System Status
    log("=" * 70)
    log("Remote Provisioning Completed Successfully!")
    log("=" * 70)
    run(f"cd {station_dir} && bash engine.sh status")


if __name__ == "__main__":
    main()

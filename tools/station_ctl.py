#!/usr/bin/env python3
"""
Colab Model Station - Local Workstation Orchestrator (Headless CLI)
Provides zero-browser provisioning, monitoring, and teardown of Colab GPU runtimes
using Google Colab CLI (`google-colab-cli`) and Tailscale mesh networking.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

REPO_DIR = Path(__file__).resolve().parents[1]
REMOTE_BOOTSTRAP = REPO_DIR / "scripts" / "remote_bootstrap.py"
MODELS_CATALOG = REPO_DIR / "configs" / "models.json"


def load_dotenv() -> None:
    """Load key-value pairs from .env into os.environ if present."""
    env_file = REPO_DIR / ".env"
    if not env_file.is_file():
        return
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val


load_dotenv()


def log(msg: str) -> None:
    print(f"[STATION] {msg}", flush=True)


def log_err(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


def check_colab_cli() -> str:
    """Verify that google-colab-cli is installed and reachable on PATH."""
    colab_bin = shutil.which("colab")
    if not colab_bin:
        log_err("Google Colab CLI ('colab') was not found on your system PATH.")
        print("\nInstallation instructions:")
        print("  1. Install via pip:   pip install google-colab-cli")
        print("     Or install uv:    uv tool install google-colab-cli")
        print("  2. Authenticate:     colab --auth=oauth2 usage\n")
        sys.exit(1)

    try:
        res = subprocess.run([colab_bin, "version"], capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            log_err(f"'colab version' failed. Output: {res.stderr.strip()}")
            sys.exit(1)
        return colab_bin
    except Exception as e:
        log_err(f"Failed to execute Colab CLI: {e}")
        sys.exit(1)


def run_colab_cmd(cmd_args: list[str], timeout: int = 1800, stream_output: bool = True) -> Tuple[int, str]:
    """Execute a colab CLI command with optional real-time streaming."""
    colab_bin = check_colab_cli()
    auth_mode = os.environ.get("COLAB_AUTH", "oauth2")
    full_cmd = [colab_bin, f"--auth={auth_mode}", *cmd_args]

    if not stream_output:
        res = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
        return res.returncode, res.stdout + res.stderr

    process = subprocess.Popen(
        full_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    collected_output: list[str] = []
    assert process.stdout is not None
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
        collected_output.append(line)

    process.stdout.close()
    returncode = process.wait()
    return returncode, "".join(collected_output)


def cmd_deploy(args: argparse.Namespace) -> None:
    """Deploy or update model station on remote Google Colab VM."""
    log("Validating environment prerequisites...")
    authkey = os.environ.get("TAILSCALE_AUTHKEY", "")
    if not authkey:
        log("Warning: TAILSCALE_AUTHKEY is not defined in your environment or .env.")

    engine = args.engine or os.environ.get("STATION_ENGINE", "diffusers")
    model = args.model or os.environ.get("STATION_MODEL", "black-forest-labs/FLUX.1-schnell")
    precision = args.precision or os.environ.get("STATION_PRECISION", "auto")

    log(f"Preparing remote bootstrap payload (Engine: {engine}, Model: {model}, Precision: {precision})...")

    env_exports = [
        f"export STATION_ENGINE='{engine}'",
        f"export STATION_MODEL='{model}'",
        f"export STATION_PRECISION='{precision}'",
    ]
    if authkey:
        env_exports.append(f"export TAILSCALE_AUTHKEY='{authkey}'")
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        env_exports.append(f"export HF_TOKEN='{hf_token}'")

    bootstrap_content = REMOTE_BOOTSTRAP.read_text(encoding="utf-8")
    remote_script = (
        "cat <<'EOF' > /tmp/remote_bootstrap.py\n"
        f"{bootstrap_content}\n"
        "EOF\n"
        + "\n".join(env_exports) + "\n"
        "python3 /tmp/remote_bootstrap.py\n"
    )

    log("Dispatching remote bootstrap sequence to Colab VM...")
    ret, _ = run_colab_cmd(["exec", remote_script])
    if ret == 0:
        log("Remote bootstrap succeeded.")
    else:
        log_err(f"Remote deployment failed with exit code {ret}.")
        sys.exit(ret)


def cmd_status(_args: argparse.Namespace) -> None:
    """Query remote runtime status and GPU utilization."""
    log("Querying Colab VM status...")
    run_colab_cmd(["status"])
    log("Inspecting remote station service status...")
    run_colab_cmd(["exec", "bash /content/colab-inference-node/engine.sh status 2>/dev/null || true"])


def cmd_stop(_args: argparse.Namespace) -> None:
    """Terminate the Google Colab VM."""
    log("Requesting Colab VM termination...")
    ret, out = run_colab_cmd(["stop"])
    if ret == 0:
        log("Colab VM stopped successfully.")
    else:
        log_err(f"Failed to stop VM: {out}")


def cmd_usage(_args: argparse.Namespace) -> None:
    """Inspect Colab compute units usage and account balance."""
    log("Fetching Colab compute unit usage...")
    run_colab_cmd(["usage"])


def main():
    parser = argparse.ArgumentParser(description="Colab Model Station Local Orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy_p = subparsers.add_parser("deploy", help="Provision and deploy engine on Colab VM")
    deploy_p.add_argument("--engine", choices=["diffusers", "comfyui"], default=None)
    deploy_p.add_argument("--model", default=None, help="Target model repository ID")
    deploy_p.add_argument("--precision", choices=["fp8", "fp16", "bf16", "auto"], default=None)
    deploy_p.set_defaults(func=cmd_deploy)

    status_p = subparsers.add_parser("status", help="Inspect remote VM and engine status")
    status_p.set_defaults(func=cmd_status)

    stop_p = subparsers.add_parser("stop", help="Stop Colab VM runtime")
    stop_p.set_defaults(func=cmd_stop)

    usage_p = subparsers.add_parser("usage", help="Display Colab compute units consumption")
    usage_p.set_defaults(func=cmd_usage)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

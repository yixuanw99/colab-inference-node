"""
ComfyUI Service Runner for colab-model-station.
Supervises headless ComfyUI process lifecycle.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "logs"
PID_FILE = LOG_DIR / "comfyui.pid"
LOG_FILE = LOG_DIR / "comfyui.log"


def get_comfyui_dir() -> Path:
    """Locate ComfyUI installation directory."""
    candidates = [
        Path("/content/ComfyUI"),
        REPO_ROOT / "engines" / "comfyui_engine" / "ComfyUI",
    ]
    for c in candidates:
        if (c / "main.py").is_file():
            return c
    return candidates[0]


def is_running(pid: int) -> bool:
    """Check if process with given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_comfyui(port: int = 8188, listen: str = "0.0.0.0", vram_mode: str = "auto") -> None:
    """Launch ComfyUI in headless background mode."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if PID_FILE.is_file():
        try:
            pid = int(PID_FILE.read_text().strip())
            if is_running(pid):
                print(f"[COMFYUI] Service is already running with PID {pid}.")
                return
        except Exception:
            pass

    comfy_dir = get_comfyui_dir()
    if not (comfy_dir / "main.py").is_file():
        print(f"[ERROR] ComfyUI not found at {comfy_dir}. Run setup first: bash engine.sh setup comfyui", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        str(comfy_dir / "main.py"),
        "--listen", listen,
        "--port", str(port),
        "--enable-cors-header", "*",
        "--preview-method", "auto",
    ]

    if vram_mode == "low":
        cmd.append("--lowvram")
    elif vram_mode == "high":
        cmd.append("--highvram")

    print(f"[COMFYUI] Launching ComfyUI daemon: {' '.join(cmd)}")
    with open(LOG_FILE, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(
            cmd,
            stdout=out,
            stderr=subprocess.STDOUT,
            cwd=str(comfy_dir),
            preexec_fn=os.setsid,
        )

    PID_FILE.write_text(str(proc.pid))
    time.sleep(3)

    if is_running(proc.pid):
        print(f"[COMFYUI] Service started successfully (PID: {proc.pid}, Port: {port}).")
    else:
        print("[ERROR] ComfyUI failed to start. Check logs:", file=sys.stderr)
        print(f"  cat {LOG_FILE}", file=sys.stderr)
        sys.exit(1)


def stop_comfyui() -> None:
    """Stop running ComfyUI process."""
    if not PID_FILE.is_file():
        print("[COMFYUI] No PID file found. ComfyUI is not running.")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        if is_running(pid):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            time.sleep(1)
            if is_running(pid):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            print(f"[COMFYUI] Stopped ComfyUI process {pid}.")
        else:
            print("[COMFYUI] Process was not running.")
    except Exception as e:
        print(f"[COMFYUI] Error terminating process: {e}")
    finally:
        if PID_FILE.is_file():
            PID_FILE.unlink()


def status_comfyui() -> None:
    """Print status of ComfyUI service."""
    if PID_FILE.is_file():
        try:
            pid = int(PID_FILE.read_text().strip())
            if is_running(pid):
                print(f"[COMFYUI] Service is RUNNING (PID: {pid}).")
                return
        except Exception:
            pass
    print("[COMFYUI] Service is STOPPED.")


def main():
    parser = argparse.ArgumentParser(description="ComfyUI Process Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_p = subparsers.add_parser("start")
    start_p.add_argument("--port", type=int, default=8188)
    start_p.add_argument("--listen", type=str, default="0.0.0.0", help="IP address to listen on (default: 0.0.0.0)")
    start_p.add_argument("--vram", choices=["auto", "low", "high"], default="auto")

    subparsers.add_parser("stop")
    subparsers.add_parser("status")

    args = parser.parse_args()
    if args.command == "start":
        start_comfyui(port=args.port, listen=args.listen, vram_mode=args.vram)
    elif args.command == "stop":
        stop_comfyui()
    elif args.command == "status":
        status_comfyui()


if __name__ == "__main__":
    main()

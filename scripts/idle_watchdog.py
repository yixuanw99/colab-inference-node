"""
Idle Watchdog Daemon for colab-model-station.
Monitors inference activity and terminates the Google Colab VM
via google.colab.runtime.unassign() when idle time exceeds threshold.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs"
PID_FILE = LOG_DIR / "watchdog.pid"
LOG_FILE = LOG_DIR / "watchdog.log"


def log(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] [WATCHDOG] {msg}"
    print(entry, flush=True)
    if LOG_FILE.parent.exists():
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")


def trigger_unassign(reason: str) -> None:
    """Terminate the Google Colab VM to protect compute units."""
    log(f"ALERT: Triggering runtime teardown due to: {reason}")
    try:
        from google.colab import runtime
        runtime.unassign()
        log("Colab runtime unassigned successfully.")
    except ImportError:
        log("Not running in Google Colab environment. Emulating unassign (stopping engines).")
        try:
            subprocess.run(["bash", str(REPO_ROOT / "engine.sh"), "diffusers", "stop"], check=False)
            subprocess.run(["bash", str(REPO_ROOT / "engine.sh"), "comfyui", "stop"], check=False)
        except Exception as e:
            log(f"Error stopping local engines: {e}")
    sys.exit(0)


def run_watchdog_loop(port: int, timeout: int, interval: int) -> None:
    """Background monitoring loop."""
    log(f"Started monitoring. Port: {port}, Idle Timeout: {timeout}s, Poll Interval: {interval}s")
    endpoint = f"http://127.0.0.1:{port}/v1/metrics"

    consecutive_errors = 0
    max_consecutive_errors = 10

    while True:
        try:
            req = urllib.request.Request(endpoint)
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                idle_seconds = data.get("idle_seconds", 0)
                consecutive_errors = 0

                log(f"Heartbeat: Service {data.get('status')} | Idle: {idle_seconds:.1f}s / {timeout}s")
                if idle_seconds >= timeout:
                    trigger_unassign(f"Idle time exceeded threshold ({idle_seconds:.1f}s >= {timeout}s)")

        except urllib.error.URLError:
            consecutive_errors += 1
            log(f"Warning: Failed to reach metrics endpoint (Consecutive failures: {consecutive_errors}/{max_consecutive_errors})")
            if consecutive_errors >= max_consecutive_errors:
                trigger_unassign(f"Inference service unreachable after {max_consecutive_errors} consecutive checks")
        except Exception as e:
            log(f"Unexpected monitoring error: {e}")

        time.sleep(interval)


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_daemon(port: int = 8000, timeout: int = 1800, interval: int = 30) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if PID_FILE.is_file():
        try:
            pid = int(PID_FILE.read_text().strip())
            if is_running(pid):
                print(f"[WATCHDOG] Already running with PID {pid}.")
                return
        except Exception:
            pass

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--port", str(port),
        "--timeout", str(timeout),
        "--interval", str(interval),
    ]

    with open(LOG_FILE, "a", encoding="utf-8") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, preexec_fn=os.setsid)

    PID_FILE.write_text(str(proc.pid))
    time.sleep(1)
    if is_running(proc.pid):
        print(f"[WATCHDOG] Started watchdog daemon (PID: {proc.pid}, Timeout: {timeout}s).")
    else:
        print("[ERROR] Watchdog daemon failed to start.", file=sys.stderr)


def stop_daemon() -> None:
    if not PID_FILE.is_file():
        print("[WATCHDOG] No watchdog PID file found.")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        if is_running(pid):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            print(f"[WATCHDOG] Stopped watchdog process {pid}.")
    except Exception as e:
        print(f"[WATCHDOG] Error stopping watchdog: {e}")
    finally:
        if PID_FILE.is_file():
            PID_FILE.unlink()


def status_daemon() -> None:
    if PID_FILE.is_file():
        try:
            pid = int(PID_FILE.read_text().strip())
            if is_running(pid):
                print(f"[WATCHDOG] Daemon is RUNNING (PID: {pid}).")
                return
        except Exception:
            pass
    print("[WATCHDOG] Daemon is STOPPED.")


def main():
    parser = argparse.ArgumentParser(description="Colab Compute Unit Protection Watchdog")
    subparsers = parser.add_subparsers(dest="command", required=True)

    start_p = subparsers.add_parser("start")
    start_p.add_argument("--port", type=int, default=8000)
    start_p.add_argument("--timeout", type=int, default=1800)
    start_p.add_argument("--interval", type=int, default=30)

    run_p = subparsers.add_parser("run")
    run_p.add_argument("--port", type=int, default=8000)
    run_p.add_argument("--timeout", type=int, default=1800)
    run_p.add_argument("--interval", type=int, default=30)

    subparsers.add_parser("stop")
    subparsers.add_parser("status")

    args = parser.parse_args()
    if args.command == "start":
        start_daemon(port=args.port, timeout=args.timeout, interval=args.interval)
    elif args.command == "run":
        run_watchdog_loop(port=args.port, timeout=args.timeout, interval=args.interval)
    elif args.command == "stop":
        stop_daemon()
    elif args.command == "status":
        status_daemon()


if __name__ == "__main__":
    main()

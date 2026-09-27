#!/usr/bin/env bash
# ==============================================================================
# Colab Model Station - Workstation & Runtime Dual Launcher
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if executing inside Google Colab environment
if [ -d "/content" ] && [ -f "$DIR/engine.sh" ]; then
    exec bash "$DIR/engine.sh" "$@"
else
    # Executing on local developer workstation (macOS / Linux / WSL)
    if [ "$1" == "generate" ]; then
        exec python3 "$DIR/tools/generate.py" "${@:2}"
    elif [ "$1" == "benchmark" ]; then
        exec python3 "$DIR/tools/benchmark.py" "${@:2}"
    elif [ "$1" == "test" ]; then
        exec python3 "$DIR/tools/test_inference.py" "${@:2}"
    else
        exec python3 "$DIR/tools/station_ctl.py" "$@"
    fi
fi

# ==============================================================================
# Colab Model Station - Local Workstation Launcher (Windows PowerShell)
# ==============================================================================
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($args.Count -eq 0) {
    python "$ScriptDir\tools\station_ctl.py" --help
    exit 0
}

$cmd = $args[0]
$remainingArgs = $args[1..($args.Count - 1)]

switch ($cmd) {
    "generate" {
        python "$ScriptDir\tools\generate.py" @remainingArgs
    }
    "benchmark" {
        python "$ScriptDir\tools\benchmark.py" @remainingArgs
    }
    "test" {
        python "$ScriptDir\tools\test_inference.py" @remainingArgs
    }
    Default {
        python "$ScriptDir\tools\station_ctl.py" @args
    }
}

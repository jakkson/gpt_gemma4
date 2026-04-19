"""Use macOS ``caffeinate`` so long ingest runs are not paused by idle/display sleep at screen lock."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable


def start_keep_awake(log: Callable[[str], None]) -> subprocess.Popen | None:
    """
    Spawn ``caffeinate -di -w <this_pid>`` so the system stays awake until this process exits.

    - ``-d`` — avoid display sleep (common when the screen locks / display turns off)
    - ``-i`` — avoid idle sleep (App Nap / idle timer)
    - ``-w`` — exit ``caffeinate`` when our Python process exits

    This does **not** override a closed laptop lid on battery, or forced sleep from
    low power; plug in for overnight runs.
    """
    if sys.platform != "darwin":
        return None
    exe = shutil.which("caffeinate")
    if not exe:
        log("[keep-awake] caffeinate not found in PATH; sleep may still pause ingest.")
        return None
    try:
        proc = subprocess.Popen(
            [exe, "-di", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log(
            "[keep-awake] caffeinate started (-di -w pid): "
            "display/idle sleep held off until ingest exits. "
            "Use --no-keep-awake to disable."
        )
        return proc
    except OSError as e:
        log(f"[keep-awake] could not start caffeinate: {e}")
        return None

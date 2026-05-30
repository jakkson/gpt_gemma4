"""Schedule macOS power-on/wake events so nightly ingest can start at 2:00 AM."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

INGEST_HOUR = 2
INGEST_MINUTE = 0
WAKE_LEAD_MINUTES = 5
WAKE_DAEMON_LABEL = "com.gptlocalgemma.photoindex.nightly-wake"


@dataclass(frozen=True)
class PowerEvent:
    kind: str
    when: datetime
    owner: str


def wake_time_for_ingest(
    *,
    ingest_hour: int = INGEST_HOUR,
    ingest_minute: int = INGEST_MINUTE,
    lead_minutes: int = WAKE_LEAD_MINUTES,
    now: datetime | None = None,
) -> datetime:
    """Next local wake time (lead_minutes before ingest)."""
    now = now or datetime.now()
    ingest_at = now.replace(hour=ingest_hour, minute=ingest_minute, second=0, microsecond=0)
    if ingest_at <= now:
        ingest_at += timedelta(days=1)
    return ingest_at - timedelta(minutes=lead_minutes)


def _parse_pmset_time(raw: str) -> datetime | None:
    raw = raw.strip()
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def parse_pmset_sched(text: str) -> list[PowerEvent]:
    events: list[PowerEvent] = []
    for line in text.splitlines():
        m = re.search(
            r"^\s*\[\d+\]\s+(wake|sleep|poweron|restart|shutdown)\s+at\s+(.+?)\s+by\s+'([^']*)'",
            line,
        )
        if not m:
            continue
        when = _parse_pmset_time(m.group(2))
        if when is None:
            continue
        events.append(PowerEvent(kind=m.group(1), when=when, owner=m.group(3)))
    return events


def read_scheduled_events() -> list[PowerEvent]:
    try:
        proc = subprocess.run(
            ["pmset", "-g", "sched"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as e:
        raise RuntimeError(f"pmset not available: {e}") from e
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "pmset -g sched failed")
    return parse_pmset_sched(proc.stdout)


def has_wake_near(
    target: datetime,
    *,
    tolerance_seconds: int = 120,
    events: list[PowerEvent] | None = None,
) -> bool:
    events = events if events is not None else read_scheduled_events()
    for event in events:
        if event.kind != "wake":
            continue
        if abs((event.when - target).total_seconds()) <= tolerance_seconds:
            return True
    return False


def schedule_wake(
    when: datetime | None = None,
    *,
    ingest_hour: int = INGEST_HOUR,
    ingest_minute: int = INGEST_MINUTE,
    lead_minutes: int = WAKE_LEAD_MINUTES,
    force: bool = False,
) -> datetime:
    """Run ``pmset schedule wake`` for the next ingest window (requires root)."""
    target = when or wake_time_for_ingest(
        ingest_hour=ingest_hour,
        ingest_minute=ingest_minute,
        lead_minutes=lead_minutes,
    )
    if not force and has_wake_near(target):
        return target

    stamp = target.strftime("%m/%d/%Y %H:%M:%S")
    proc = subprocess.run(
        ["pmset", "schedule", "wake", stamp],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(err or f"pmset schedule wake failed for {stamp}")
    return target


def _log(msg: str) -> None:
    print(msg, flush=True)


def cmd_schedule(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Schedule the next macOS wake before nightly ingest.")
    p.add_argument("--force", action="store_true", help="Schedule even if a nearby wake exists.")
    p.add_argument("--ingest-hour", type=int, default=INGEST_HOUR)
    p.add_argument("--ingest-minute", type=int, default=INGEST_MINUTE)
    p.add_argument("--lead-minutes", type=int, default=WAKE_LEAD_MINUTES)
    args = p.parse_args(argv)
    try:
        target = schedule_wake(
            ingest_hour=args.ingest_hour,
            ingest_minute=args.ingest_minute,
            lead_minutes=args.lead_minutes,
            force=args.force,
        )
    except RuntimeError as e:
        _log(f"[nightly-wake error] {e}")
        if "Must be run as root" in str(e):
            _log("[nightly-wake hint] Run via the root LaunchDaemon or: sudo .venv/bin/python -m photo_index.nightly_wake schedule")
        return 1
    ingest_at = target + timedelta(minutes=args.lead_minutes)
    _log(
        f"[nightly-wake] scheduled wake at {target.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(ingest at {ingest_at.strftime('%H:%M')})"
    )
    return 0


def cmd_status(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Show nightly wake schedule status.")
    p.add_argument("--ingest-hour", type=int, default=INGEST_HOUR)
    p.add_argument("--ingest-minute", type=int, default=INGEST_MINUTE)
    p.add_argument("--lead-minutes", type=int, default=WAKE_LEAD_MINUTES)
    args = p.parse_args(argv)

    target = wake_time_for_ingest(
        ingest_hour=args.ingest_hour,
        ingest_minute=args.ingest_minute,
        lead_minutes=args.lead_minutes,
    )
    ingest_at = target + timedelta(minutes=args.lead_minutes)
    _log(f"[nightly-wake] next ingest window: {ingest_at.strftime('%Y-%m-%d %H:%M')} local")
    _log(f"[nightly-wake] desired wake:       {target.strftime('%Y-%m-%d %H:%M:%S')} local")

    try:
        events = read_scheduled_events()
    except RuntimeError as e:
        _log(f"[nightly-wake error] {e}")
        return 1

    wakes = [e for e in events if e.kind == "wake"]
    if not wakes:
        _log("[nightly-wake] no wake events in pmset schedule (install nightly wake daemon)")
        return 1

    _log("[nightly-wake] pmset wake events:")
    for event in sorted(wakes, key=lambda e: e.when):
        _log(f"  - {event.when.strftime('%Y-%m-%d %H:%M:%S')} ({event.owner or 'unknown'})")

    if has_wake_near(target, events=events):
        _log("[nightly-wake] ok: a wake is scheduled near the next ingest window")
        return 0

    _log("[nightly-wake] warn: no wake found within 2 minutes of the desired time")
    _log("[nightly-wake] hint: ./install_photo_nightly_launchd.sh  (re-install wake scheduler)")
    return 1


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        p = argparse.ArgumentParser(
            description="macOS wake scheduling for photo_index nightly ingest.",
        )
        p.add_argument("command", choices=["schedule", "status"])
        p.print_help()
        raise SystemExit(0)

    command = argv[0]
    rest = argv[1:]
    if command == "schedule":
        sys.exit(cmd_schedule(rest))
    if command == "status":
        sys.exit(cmd_status(rest))
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()

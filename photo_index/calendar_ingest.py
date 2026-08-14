#!/usr/bin/env python3
"""Ingest Apple Calendar (iCal) events into the personal index.

Reads the macOS Calendar store and writes one index row per event so the RAG
assistant can answer "what's my next appointment", "what's on my calendar next
week", "when did I see X", etc.

Which calendars are ingested (per the owner's rule "all personal calendars
except subscribed and holidays; holidays only for the next 365 days"):

  * PERSONAL  (iCloud / Exchange / birthdays / manual)  -> ingested in full
  * HOLIDAY   (title contains "holiday")                -> only events whose
                                                            start falls in the
                                                            next 365 days
  * SUBSCRIBED feeds (webcal `subcal_url`, non-holiday)  -> skipped
  * REMINDERS (store type 6)                             -> skipped
  * "Found in Mail" / "Found in Natural Language"        -> skipped (these are
        auto-detected from mail and would duplicate the mail ingest)
  * Known noise feeds ("Phases of the Moon", …)          -> skipped

Apple stores event times as seconds since 2001-01-01 (the "Apple epoch");
we add 978307200 to convert to Unix time.

The Calendar DB is copied to a temp file before reading so a running Calendar
app can't lock us out and we never touch the live store.
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from photo_index.ingest_lock import global_ingest_lock
from photo_index.store import commit_ingest, connect, init_schema, upsert_photo

# Apple absolute time -> Unix time offset (seconds from 2001-01-01 to 1970-01-01).
_APPLE_EPOCH = 978307200

# Default location of the macOS Calendar store.
_DEFAULT_CAL_DB = (
    Path.home()
    / "Library/Group Containers/group.com.apple.calendar/Calendar.sqlitedb"
)

# Store.type values seen in the wild.
_STORE_REMINDERS = 6  # skip: reminders, not calendar events

# Calendars whose titles mark them as non-personal even though Apple does not
# flag them as subscribed feeds (case-insensitive substring match).
_SKIP_TITLE_SUBSTRINGS = (
    "phases of the moon",
    "moon phase",
)

# Auto-generated calendars that mirror content ingested elsewhere.
_SKIP_TITLE_EXACT = {
    "found in mail",
    "found in natural language",
}

# Personal-event date sanity window: drop obviously bogus timestamps (year 1604
# placeholders, absurd future) while keeping decades of real history.
_MIN_YEAR = 2005
_MAX_FUTURE_YEARS = 5


def _log(msg: str) -> None:
    print(f"[calendar] {msg}", file=sys.stderr, flush=True)


def _apple_to_dt(apple_secs: float) -> datetime | None:
    """Convert an Apple-epoch timestamp to a timezone-aware UTC datetime."""
    if apple_secs is None:
        return None
    try:
        return datetime.fromtimestamp(float(apple_secs) + _APPLE_EPOCH, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def _copy_calendar_db(src: Path, tmpdir: Path) -> Path:
    """Copy the Calendar store (+ WAL/SHM sidecars) so we read a stable snapshot."""
    dst = tmpdir / src.name
    shutil.copy2(src, dst)
    for suffix in ("-wal", "-shm"):
        side = src.with_name(src.name + suffix)
        if side.exists():
            shutil.copy2(side, tmpdir / side.name)
    return dst


def _classify_calendar(title: str, store_type: int | None, subcal_url) -> str:
    """Return 'personal', 'holiday', or 'skip' for a calendar."""
    title_l = (title or "").strip().lower()

    if store_type == _STORE_REMINDERS:
        return "skip"
    if title_l in _SKIP_TITLE_EXACT:
        return "skip"
    if any(s in title_l for s in _SKIP_TITLE_SUBSTRINGS):
        return "skip"

    is_holiday = "holiday" in title_l
    if is_holiday:
        return "holiday"

    is_subscribed = subcal_url is not None and str(subcal_url).strip() != ""
    if is_subscribed:
        return "skip"

    return "personal"


def _load_calendars(cal: sqlite3.Connection) -> dict[int, dict]:
    """Map Calendar.ROWID -> {title, klass} for the calendars we ingest."""
    store_types: dict[int, int] = {}
    for rowid, stype in cal.execute("SELECT ROWID, type FROM Store"):
        store_types[rowid] = stype

    out: dict[int, dict] = {}
    rows = cal.execute(
        "SELECT ROWID, title, store_id, subcal_url FROM Calendar"
    ).fetchall()
    for rowid, title, store_id, subcal_url in rows:
        klass = _classify_calendar(title, store_types.get(store_id), subcal_url)
        out[rowid] = {"title": title or "(untitled)", "klass": klass}
    return out


def _load_locations(cal: sqlite3.Connection) -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        rows = cal.execute("SELECT ROWID, title, address FROM Location").fetchall()
    except sqlite3.OperationalError:
        return out
    for rowid, title, address in rows:
        parts = [p for p in (title, address) if p and p.strip()]
        if parts:
            # Avoid "Home, Home" style duplication.
            if len(parts) == 2 and parts[0].strip() == parts[1].strip():
                parts = parts[:1]
            out[rowid] = ", ".join(p.strip() for p in parts)
    return out


def _event_uuid(unique_id, rowid: int, start_apple) -> str:
    base = (unique_id or f"row{rowid}").strip() or f"row{rowid}"
    start_tag = int(start_apple) if start_apple is not None else 0
    return f"cal:{base}:{start_tag}"


def _render_when(start_dt: datetime | None, end_dt: datetime | None, all_day: int) -> str:
    if start_dt is None:
        return "(unknown time)"
    if all_day:
        if end_dt and end_dt.date() > start_dt.date():
            last = (end_dt - timedelta(days=1)).date()
            if last > start_dt.date():
                return f"{start_dt:%Y-%m-%d} to {last:%Y-%m-%d} (all day)"
        return f"{start_dt:%Y-%m-%d} (all day)"
    s = f"{start_dt:%Y-%m-%d %H:%M}"
    if end_dt:
        if end_dt.date() == start_dt.date():
            return f"{s}-{end_dt:%H:%M}"
        return f"{s} to {end_dt:%Y-%m-%d %H:%M}"
    return s


def run_calendar_ingest(
    db_path: Path,
    cal_db_path: Path,
    *,
    holiday_days: int = 365,
    progress_every: int = 200,
    dry_run: bool = False,
) -> dict:
    if not cal_db_path.exists():
        _log(f"Calendar store not found: {cal_db_path}")
        return {"indexed": 0, "skipped_unchanged": 0, "skipped_filtered": 0, "errors": 1}

    now = datetime.now(timezone.utc)
    holiday_cutoff = now + timedelta(days=holiday_days)
    min_dt = datetime(_MIN_YEAR, 1, 1, tzinfo=timezone.utc)
    max_dt = datetime(now.year + _MAX_FUTURE_YEARS, 12, 31, tzinfo=timezone.utc)

    indexed = 0
    skipped_unchanged = 0
    skipped_filtered = 0
    errors = 0

    with tempfile.TemporaryDirectory(prefix="cal_ingest_") as td:
        tmp = Path(td)
        snapshot = _copy_calendar_db(cal_db_path, tmp)
        cal = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
        cal.row_factory = None

        calendars = _load_calendars(cal)
        locations = _load_locations(cal)

        ingest_cals = {cid: c for cid, c in calendars.items() if c["klass"] != "skip"}
        n_personal = sum(1 for c in ingest_cals.values() if c["klass"] == "personal")
        n_holiday = sum(1 for c in ingest_cals.values() if c["klass"] == "holiday")
        _log(
            f"{len(ingest_cals)} calendars to ingest "
            f"({n_personal} personal, {n_holiday} holiday); "
            f"{len(calendars) - len(ingest_cals)} skipped"
        )

        conn = connect(db_path)
        init_schema(conn)

        # Pre-load existing calendar rows' change-tag so we can skip unchanged
        # events without a per-event SELECT.
        existing_tag: dict[str, str] = {}
        for uuid, vlm in conn.execute(
            "SELECT uuid, vlm_text FROM photo_meta WHERE uuid LIKE 'cal:%'"
        ):
            existing_tag[uuid] = vlm or ""

        rows = cal.execute(
            """
            SELECT ROWID, summary, description, start_date, end_date,
                   all_day, calendar_id, location_id, last_modified,
                   unique_identifier
            FROM CalendarItem
            WHERE calendar_id IS NOT NULL AND start_date IS NOT NULL
            """
        )

        seen = 0
        for (
            rowid,
            summary,
            description,
            start_apple,
            end_apple,
            all_day,
            calendar_id,
            location_id,
            last_modified,
            unique_id,
        ) in rows:
            cal_info = ingest_cals.get(calendar_id)
            if cal_info is None:
                continue  # skipped calendar

            start_dt = _apple_to_dt(start_apple)
            if start_dt is None:
                skipped_filtered += 1
                continue

            klass = cal_info["klass"]
            if klass == "holiday":
                if not (now <= start_dt <= holiday_cutoff):
                    skipped_filtered += 1
                    continue
            else:  # personal
                if not (min_dt <= start_dt <= max_dt):
                    skipped_filtered += 1
                    continue

            title = (summary or "").strip() or "(no title)"
            cal_name = cal_info["title"]
            end_dt = _apple_to_dt(end_apple)
            where = locations.get(location_id, "")
            when = _render_when(start_dt, end_dt, all_day)

            uuid = _event_uuid(unique_id, rowid, start_apple)
            # Change tag embeds last_modified + start so an edited/moved event
            # re-embeds, an untouched one is skipped.
            tag = f"cal-event mod={last_modified} start={start_apple}"

            prev = existing_tag.get(uuid)
            if prev is not None and prev.startswith(tag):
                skipped_unchanged += 1
                continue

            lines = [
                f"Calendar event: {title}",
                f"When: {when}",
            ]
            if where:
                lines.append(f"Where: {where}")
            lines.append(f"Calendar: {cal_name}")
            notes = (description or "").strip()
            if notes:
                lines.append(f"Notes: {notes}")
            ocr_text = "\n".join(lines)

            filename = f"{start_dt:%Y-%m-%d} | {title} [{cal_name}]"
            date_iso = start_dt.astimezone(timezone.utc).isoformat()

            if not dry_run:
                try:
                    upsert_photo(
                        conn,
                        uuid=uuid,
                        filename=filename,
                        date_iso=date_iso,
                        ocr_text=ocr_text,
                        vlm_text=tag,
                        image_path_used="",
                        open_url="",
                        commit=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    _log(f"error on event {rowid}: {exc}")
                    continue

            indexed += 1
            seen += 1
            if progress_every and seen % progress_every == 0:
                if not dry_run:
                    conn.commit()
                _log(f"  … {indexed} events indexed")

        cal.close()
        if not dry_run:
            commit_ingest(conn)
        conn.close()

    _log(
        f"done: indexed={indexed} skipped_unchanged={skipped_unchanged} "
        f"skipped_filtered={skipped_filtered} errors={errors}"
        + (" (dry run — nothing written)" if dry_run else "")
    )
    return {
        "indexed": indexed,
        "skipped_unchanged": skipped_unchanged,
        "skipped_filtered": skipped_filtered,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Ingest Apple Calendar events.")
    ap.add_argument("--db", type=Path, required=True, help="photo_index sqlite path")
    ap.add_argument(
        "--calendar-db",
        type=Path,
        default=_DEFAULT_CAL_DB,
        help="path to Calendar.sqlitedb",
    )
    ap.add_argument("--holiday-days", type=int, default=365)
    ap.add_argument("--progress-every", type=int, default=200)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="classify and count events but write nothing",
    )
    ap.add_argument(
        "--list-calendars",
        action="store_true",
        help="print each calendar and how it is classified, then exit",
    )
    args = ap.parse_args(argv)

    if args.list_calendars:
        with tempfile.TemporaryDirectory(prefix="cal_list_") as td:
            snap = _copy_calendar_db(args.calendar_db, Path(td))
            cal = sqlite3.connect(f"file:{snap}?mode=ro", uri=True)
            for cid, info in sorted(
                _load_calendars(cal).items(), key=lambda kv: kv[1]["klass"]
            ):
                print(f"  {info['klass']:9s}  {info['title']}")
            cal.close()
        return

    with global_ingest_lock():
        run_calendar_ingest(
            args.db,
            args.calendar_db,
            holiday_days=args.holiday_days,
            progress_every=args.progress_every,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()

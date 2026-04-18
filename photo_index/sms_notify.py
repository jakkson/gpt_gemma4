"""Optional SMS notifications for photo index runs (Twilio)."""

from __future__ import annotations

import os
import traceback
from typing import Any

# Default destination (E.164). Override with PHOTO_INDEX_SMS_TO.
_DEFAULT_SMS_TO = "+14158770063"


def _sms_enabled() -> bool:
    if os.environ.get("PHOTO_INDEX_SMS", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return False
    return bool(
        os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
        and os.environ.get("TWILIO_AUTH_TOKEN", "").strip()
        and os.environ.get("TWILIO_FROM", "").strip()
    )


def _sms_to() -> str:
    return os.environ.get("PHOTO_INDEX_SMS_TO", _DEFAULT_SMS_TO).strip()


def send_sms(body: str) -> bool:
    """
    Send one SMS via Twilio. Returns True if sent, False if SMS disabled or misconfigured.

    Required env when PHOTO_INDEX_SMS=1:
      TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM (your Twilio number, E.164)

    Optional:
      PHOTO_INDEX_SMS_TO — destination (default +14158770063)
    """
    if not _sms_enabled():
        return False
    body = body.strip()
    if len(body) > 1500:
        body = body[:1497] + "..."

    try:
        from twilio.rest import Client
    except ImportError as e:
        raise RuntimeError(
            "SMS enabled but twilio is not installed. Run: pip install twilio"
        ) from e

    client = Client(
        os.environ["TWILIO_ACCOUNT_SID"].strip(),
        os.environ["TWILIO_AUTH_TOKEN"].strip(),
    )
    client.messages.create(
        body=body,
        from_=os.environ["TWILIO_FROM"].strip(),
        to=_sms_to(),
    )
    return True


def notify_ingest_success(stats: dict[str, Any]) -> None:
    msg = (
        f"gpt-local-gemma photo_index COMPLETE. "
        f"indexed={stats.get('ok', 0)} "
        f"skipped_no_file={stats.get('skip_no_path', 0)} "
        f"skipped_dup={stats.get('skip_dup', 0)} "
        f"vlm_errors={stats.get('errors', 0)} "
        f"candidates={stats.get('total', 0)} "
        f"time_s={stats.get('elapsed', 0):.1f}"
    )
    try:
        if send_sms(msg):
            print("[sms] Sent completion SMS.", flush=True)
    except Exception as e:
        print(f"[sms] Failed to send completion SMS: {e}", flush=True)


def notify_ingest_failure(exc: BaseException) -> None:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    msg = f"gpt-local-gemma photo_index FAILED: {exc!r}\n{tb}"[:1500]
    try:
        if send_sms(msg):
            print("[sms] Sent failure SMS.", flush=True)
    except Exception as e:
        print(f"[sms] Failed to send failure SMS: {e}", flush=True)

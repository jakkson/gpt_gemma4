#!/usr/bin/env bash
# Double-clickable launcher for the Mail Rule Dropper app.
cd "$(dirname "$0")/.."
exec .venv/bin/python scripts/mail_dropper.py

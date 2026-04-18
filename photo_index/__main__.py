"""python -m photo_index ingest ... | search ..."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m photo_index ingest [args] | python -m photo_index search QUESTION", file=sys.stderr)
        sys.exit(2)
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd == "ingest":
        from photo_index.ingest import main as ingest_main

        ingest_main(rest)
    elif cmd == "search":
        from photo_index.search_cli import main as search_main

        search_main(rest)
    else:
        print("Unknown command; use ingest or search.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

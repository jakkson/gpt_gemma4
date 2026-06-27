#!/usr/bin/env python3
"""Audit a folder before ingest: file-type "target list" + offload candidates.

Walks a folder tree and reports what's inside, grouped by file extension and by
high-level category, so you can offload bulky/non-ingestable items (video, audio,
raw media, archives, model weights) before indexing the rest.

Produces:
  1. A per-extension CSV  (extension, count, total_size, avg_size, category,
     OFFLOAD column)  ordered by total size — the space hogs float to the top.
  2. A printed category summary (Video / Audio / Images / Documents / Archives /
     Code+Data / Other) with totals, so you can decide in broad strokes.

Nothing is moved or deleted — this is read-only analysis. To act on it, mark the
OFFLOAD column (or pick categories) and use --emit-move to generate a dry-run
`mv` script into a holding folder.

Usage:
  python scripts/folder_audit.py /path/to/folder
  python scripts/folder_audit.py /path/to/folder --csv /tmp/audit.csv
  python scripts/folder_audit.py /path/to/folder --emit-move ~/Offload > move.sh
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

# Extension -> category. Lowercase, no leading dot. Anything unlisted -> "Other".
CATEGORY_EXT = {
    "Video": {"mov", "mp4", "m4v", "avi", "mkv", "wmv", "flv", "webm", "mpg",
              "mpeg", "3gp", "mts", "m2ts", "prproj", "fcpbundle", "mogrt"},
    "Audio": {"wav", "mp3", "aif", "aiff", "m4a", "flac", "aac", "ogg", "wma",
              "mid", "midi", "logicx", "band", "als", "flp"},
    "Images": {"jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif", "heic", "webp",
               "svg", "ico"},
    "RawMedia": {"psd", "ai", "raw", "cr2", "cr3", "nef", "arw", "dng", "orf",
                 "rw2", "indd", "eps", "sketch", "xcf", "aep", "aegraphic"},
    "Documents": {"pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf",
                  "txt", "md", "pages", "numbers", "key", "csv", "odt", "epub"},
    "Archives": {"zip", "tar", "gz", "tgz", "bz2", "7z", "rar", "dmg", "pkg",
                 "iso", "cdr"},
    "CodeData": {"py", "js", "ts", "json", "xml", "html", "css", "sql", "sh",
                 "ipynb", "pth", "ckpt", "safetensors", "bin", "tar", "h5",
                 "pt", "onnx", "gguf", "mlx", "db", "sqlite"},
}
EXT_CATEGORY = {ext: cat for cat, exts in CATEGORY_EXT.items() for ext in exts}

# Categories that are usually NOT worth ingesting into a text/RAG index.
OFFLOAD_DEFAULT = {"Video", "Audio", "RawMedia", "Archives", "CodeData"}


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f} {unit}" if unit != "B" else f"{int(f)} B"
        f /= 1024
    return f"{f:.1f} TB"


def category_for(ext: str) -> str:
    return EXT_CATEGORY.get(ext.lower(), "Other")


def audit(root: Path):
    """Return (ext_stats, skipped_dirs). ext_stats: ext -> [count, total_bytes]."""
    ext_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for dirpath, dirnames, filenames in os.walk(root):
        # skip noise dirs that bloat counts and are never ingest targets
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "node_modules", "__pycache__",
                                    ".venv", "venv", ".Trash")]
        for name in filenames:
            if name.startswith("._"):  # AppleDouble resource forks
                continue
            ext = Path(name).suffix.lower().lstrip(".") or "(no ext)"
            try:
                size = (Path(dirpath) / name).stat().st_size
            except OSError:
                size = 0
            s = ext_stats[ext]
            s[0] += 1
            s[1] += size
    return ext_stats


def write_csv(ext_stats, out_path: Path):
    rows = sorted(ext_stats.items(), key=lambda kv: kv[1][1], reverse=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "extension", "count", "total_size", "total_bytes",
                    "avg_size", "category", "OFFLOAD (put x)"])
        for i, (ext, (cnt, total)) in enumerate(rows, 1):
            avg = total // cnt if cnt else 0
            cat = category_for(ext)
            pre = "x" if cat in OFFLOAD_DEFAULT else ""
            w.writerow([i, ext, cnt, human(total), total, human(avg), cat, pre])
    return rows


def category_summary(ext_stats):
    cat_stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ext, (cnt, total) in ext_stats.items():
        c = cat_stats[category_for(ext)]
        c[0] += cnt
        c[1] += total
    return sorted(cat_stats.items(), key=lambda kv: kv[1][1], reverse=True)


def emit_move(root: Path, ext_stats, holding: Path):
    """Print a dry-run-friendly shell script moving OFFLOAD_DEFAULT files."""
    offload_exts = {ext for ext in ext_stats
                    if category_for(ext) in OFFLOAD_DEFAULT and ext != "(no ext)"}
    print("#!/usr/bin/env bash")
    print("# Review, then run. Files move to a holding folder (reversible).")
    print(f'HOLD={holding!s}')
    print('mkdir -p "$HOLD"')
    for ext in sorted(offload_exts):
        print(f'# {ext}: {ext_stats[ext][0]} files, {human(ext_stats[ext][1])}')
        print(f'find {root!s} -type f -iname "*.{ext}" -print0 | '
              f'xargs -0 -I{{}} echo mv {{}} "$HOLD/"   # drop the echo to execute')


def main(argv=None):
    p = argparse.ArgumentParser(description="Pre-ingest folder audit / target list.")
    p.add_argument("folder", help="Folder to audit (recursively).")
    p.add_argument("--csv", default=None, help="CSV output path (default: <folder>_audit.csv in cwd).")
    p.add_argument("--emit-move", metavar="HOLDING_DIR",
                   help="Print a dry-run mv script offloading bulky categories.")
    args = p.parse_args(argv)

    root = Path(args.folder).expanduser()
    if not root.is_dir():
        print(f"Not a folder: {root}", file=sys.stderr)
        sys.exit(1)

    ext_stats = audit(root)
    if not ext_stats:
        print("No files found.")
        return

    if args.emit_move:
        emit_move(root, ext_stats, Path(args.emit_move).expanduser())
        return

    out = Path(args.csv) if args.csv else Path.cwd() / f"{root.name}_audit.csv"
    write_csv(ext_stats, out)

    total_files = sum(c for c, _ in ext_stats.values())
    total_bytes = sum(b for _, b in ext_stats.values())
    print(f"Audited: {root}")
    print(f"Total: {total_files:,} files, {human(total_bytes)}")
    print()
    print(f"{'CATEGORY':<12}{'FILES':>10}{'SIZE':>12}   offload-default")
    print("-" * 52)
    for cat, (cnt, total) in category_summary(ext_stats):
        mark = "offload" if cat in OFFLOAD_DEFAULT else "keep"
        print(f"{cat:<12}{cnt:>10,}{human(total):>12}   {mark}")
    print()
    print(f"Target-list CSV written: {out}")
    print("Mark the OFFLOAD column (pre-filled for bulky categories), or rerun")
    print(f"with --emit-move <dir> to generate a dry-run mv script.")


if __name__ == "__main__":
    main()

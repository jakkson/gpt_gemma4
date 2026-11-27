#!/usr/bin/env python3
"""Replace an updated (MEAP or any versioned) book in the index, safely.

Point it at the NEW book file(s); for each it finds every OLD version/format of
the same title across the ingest roots (~/LLM_Books AND ~/Dropbox/Documents) and
in the DB, shows a plan, and — after you confirm — purges the old ones (files
stashed, not deleted; rows removed by exact path), drops in the new file, then
re-ingests + embeds and restarts the search app.

Safe by design: rolling DB backup first, old files moved (reversible), rows
deleted by EXACT image_path_used, and nothing destructive happens before you
type 'yes'.

Usage:
  .venv/bin/python scripts/replace_book.py /path/to/New_Book_v5_MEAP.epub [more...]
"""
from __future__ import annotations
import os, re, shutil, sqlite3, subprocess, sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "photo_index.sqlite"
BOOKS = Path.home() / "LLM_Books"
ROOTS = [BOOKS, Path.home() / "Dropbox" / "Documents"]      # where books get indexed
STASH = Path.home() / "Downloads" / "_replaced_books"
PY = REPO / ".venv" / "bin" / "python"

_VER = re.compile(r"_+v\d+(?:[._-]?meap)?$", re.I)          # _v5 / __v6 / _v5_MEAP
_MEAP = re.compile(r"_+meap$", re.I)


def base_title(name: str) -> str:
    """Title with the version/MEAP suffix and extension stripped, so v4 and v5 of
    the same book map to the same base ('Designing_AI_Systems')."""
    stem = Path(name).stem
    prev = None
    while stem != prev:                                     # strip repeatedly
        prev = stem
        stem = re.sub(r"\s+copy(\s*\d+)?$", "", stem, flags=re.I)   # Finder "… copy"
        stem = _VER.sub("", stem)
        stem = _MEAP.sub("", stem)
    return stem


def old_versions_in_db(conn: sqlite3.Connection, base: str, new_name: str):
    """[(path, chunks)] for indexed files of the same base title, excluding the new
    file. Matches by exact recomputed base (LIKE is only a pre-filter)."""
    esc = base.replace("\\", "\\\\").replace("_", "\\_").replace("%", "\\%")
    rows = conn.execute(
        "SELECT image_path_used, COUNT(*) FROM photo_meta "
        "WHERE filename LIKE ? ESCAPE '\\' GROUP BY image_path_used",
        (esc + "%",),
    ).fetchall()
    out = []
    for path, n in rows:
        if not path:
            continue
        fn = Path(path).name
        if base_title(fn) == base and fn != new_name:
            out.append((path, n))
    return out


def old_files_on_disk(base: str, new_name: str):
    hits = []
    for root in ROOTS:
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file() and base_title(p.name) == base and p.name != new_name:
                hits.append(p)
    return hits


def main() -> int:
    news = [Path(a).expanduser().resolve() for a in sys.argv[1:]]
    news = [p for p in news if p.is_file()]
    if not news:
        print("Pass one or more new book files. Nothing to do.")
        return 1

    # Build the full plan across all the new files first.
    conn = sqlite3.connect(DB)
    plan = []   # (new_path, base, [old_db_paths], [old_files])
    for np in news:
        base = base_title(np.name)
        plan.append((np, base, old_versions_in_db(conn, base, np.name),
                     old_files_on_disk(base, np.name)))
    conn.close()

    print("\n===== BOOK REPLACE PLAN =====")
    total_rows = 0
    for np, base, dbold, fsold in plan:
        print(f"\nNEW: {np.name}")
        print(f"  title: {base}")
        if dbold:
            for path, n in dbold:
                total_rows += n
                loc = "LLM_Books" if "LLM_Books" in path else (
                      "Dropbox/Documents" if "Dropbox/Documents" in path else path)
                print(f"  REMOVE old (index): {Path(path).name}  [{n} chunks, {loc}]")
        else:
            print("  (no old version in the index — new title, just adding)")
        for f in fsold:
            print(f"  STASH old file: {f}")
    print(f"\nTotal old chunks to delete: {total_rows}")
    print("Old files are MOVED to " + str(STASH) + " (reversible). DB backed up first.")
    ans = input("\nType 'yes' to proceed: ").strip().lower()
    if ans != "yes":
        print("Aborted — nothing changed.")
        return 1

    # 1) rolling DB backup (single file, overwritten each run).
    bak = REPO / "photo_index.book-swap.bak.sqlite"
    print(f"\n[1/5] backing up DB -> {bak.name} ...")
    subprocess.run(["/usr/bin/sqlite3", str(DB), f".backup '{bak}'"], check=True)

    # 2) stash old files + 3) delete old rows.
    STASH.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    for np, base, dbold, fsold in plan:
        for f in fsold:
            try:
                shutil.move(str(f), str(STASH / f.name))
                print(f"[2/5] stashed {f.name}")
            except Exception as e:
                print(f"      warn: could not stash {f}: {e}")
        for path, _ in dbold:
            conn.execute("DELETE FROM photo_meta WHERE image_path_used = ?", (path,))
        # 4) place the new file into the library.
        dest = BOOKS / np.name
        shutil.copy2(str(np), str(dest))
        print(f"[3/5] placed {np.name} in LLM_Books")
    conn.commit()
    conn.close()

    # 5) re-ingest + embed, then restart the app.
    env = {**os.environ, "PHOTO_INDEX_LLM_BACKEND": "openai",
           "PHOTO_INDEX_LLM_BASE_URL": "http://127.0.0.1:1234/v1"}
    print("[4/5] ingesting new book(s) + embedding (this can take a few minutes) ...")
    subprocess.run([str(PY), "-m", "photo_index.documents_ingest",
                    "--root", str(BOOKS), "--index-media-filenames"],
                   cwd=str(REPO), env=env)
    subprocess.run([str(PY), "-m", "photo_index.embed_index"], cwd=str(REPO), env=env)

    print("[5/5] restarting the search app ...")
    subprocess.run(["pkill", "-f", "photo_index.gradio_app"])
    import time; time.sleep(3)
    subprocess.run(["pkill", "-9", "-f", "photo_index.gradio_app"])
    subprocess.Popen(["bash", str(REPO / "start_search.sh")], cwd=str(REPO),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)

    # Verify.
    conn = sqlite3.connect(DB)
    print("\n===== DONE — new chunk counts =====")
    for np, base, _, _ in plan:
        n = conn.execute("SELECT COUNT(*) FROM photo_meta WHERE image_path_used = ?",
                         (str(BOOKS / np.name),)).fetchone()[0]
        left = conn.execute(
            "SELECT COUNT(*) FROM photo_meta WHERE filename LIKE ? ESCAPE '\\' "
            "AND image_path_used <> ?",
            (base.replace("_", "\\_") + "%", str(BOOKS / np.name))).fetchone()[0]
        print(f"  {np.name}: {n} chunks   (other '{base}' rows left: {left})")
    conn.close()
    print(f"\nOld files are in {STASH} — delete when you're happy. Backup: {bak.name}")
    print("The search app is restarting; give it ~30s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Qwen-Image Studio — standalone local text-to-image for the MacBook (MLX/mflux).

Deliberately SEPARATE from the RAG search app (photo_index/gradio_app.py):
  * Memory: Qwen-Image needs ~20 GB resident; it cannot coexist with the ~103 GB
    Qwen3-235B LLM on a 128 GB Mac — unload the LLM in LM Studio before generating.
  * Runtime: image gen runs through mflux/MLX (its own process + venv), not the
    LM Studio OpenAI API the RAG app uses.
  * It must NEVER run on the 32 GB mini — it would OOM/panic alongside the 30B.

Backend: the `mflux-generate-qwen-2.1` CLI (Qwen-Image 2.1). The FIRST generate
downloads ~33 GB of weights from Hugging Face and quantizes to the chosen bits.

Usage (on the MacBook):
  image_studio/setup.sh                       # one-time: venv + mflux + gradio
  image_studio/.venv/bin/python image_studio/app.py
  open http://127.0.0.1:7863
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

OUT_DIR = Path(os.environ.get("IMAGE_STUDIO_OUT", str(Path.home() / "Generated Images")))
# Which mflux command to use. Override if your mflux version names it differently
# (e.g. the base model is `mflux-generate-qwen`).
MFLUX_CMD = os.environ.get("MFLUX_QWEN_CMD", "mflux-generate-qwen-2.1")
# LM Studio CLI, used by the "free the LLM" button to unload the big model so its
# memory is available for image generation.
LMS_CMD = os.environ.get("LMS_CMD", "lms")
PORT = int(os.environ.get("IMAGE_STUDIO_PORT", "7863"))
DEFAULT_QUANT = os.environ.get("IMAGE_STUDIO_QUANT", "8")
# Flags not in the confirmed core set; dropped on an "unrecognized argument" retry
# so the app still works across mflux versions.
_ARG_ERR_HINTS = ("unrecognized arguments", "no such option", "unexpected extra",
                  "error: argument", "invalid choice")


def _newest_png(since: float, *dirs) -> Path | None:
    best, best_m = None, since
    for d in dirs:
        p = Path(d) if d else None
        if not p or not p.is_dir():
            continue
        for f in p.glob("*.png"):
            try:
                m = f.stat().st_mtime
            except OSError:
                continue
            if m >= best_m:
                best, best_m = f, m
    return best


def _resolve_cmd() -> str | None:
    """mflux's CLI lives next to this venv's python; PATH isn't set when launched
    via ./.venv/bin/python, so look there first, then fall back to PATH."""
    cand = Path(sys.executable).parent / MFLUX_CMD
    if cand.exists():
        return str(cand)
    return shutil.which(MFLUX_CMD)


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=3600)


def free_llm():
    """Unload ALL LM Studio models so their memory frees up for image generation.
    Model-agnostic on purpose — it doesn't matter which LLM is loaded."""
    exe = shutil.which(LMS_CMD)
    if not exe:
        return ("LM Studio CLI `lms` not on PATH. One-time: open LM Studio and run its "
                "`lms bootstrap`, or just eject the model manually in LM Studio "
                "(🖥️ Developer tab → eject / power icon) before generating.")
    try:
        p = subprocess.run([exe, "unload", "--all"], capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return f"Failed to run `lms unload --all`: {e}"
    msg = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    if p.returncode == 0:
        return f"✅ Unloaded LM Studio models — memory freed for image gen.\n{msg}"
    return f"`lms unload --all` exit {p.returncode}:\n{msg}"


def generate(prompt, negative, width, height, steps, seed, quant, guidance,
             progress=gr.Progress()):
    prompt = (prompt or "").strip()
    if not prompt:
        return None, "Enter a prompt first."
    exe = _resolve_cmd()
    if not exe:
        return None, (f"`{MFLUX_CMD}` not found. Run image_studio/setup.sh first, or set "
                      "MFLUX_QWEN_CMD to the mflux Qwen command your version installs "
                      "(e.g. mflux-generate-qwen).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seed = int(seed) if seed is not None else -1
    if seed < 0:
        seed = int(time.time()) % 2_000_000_000
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = "".join(c for c in prompt[:40] if c.isalnum() or c in " -_").strip().replace(" ", "_")
    out = OUT_DIR / f"{ts}_{slug or 'image'}_seed{seed}.png"

    core = [exe, "--prompt", prompt, "--steps", str(int(steps)), "--seed", str(seed),
            "--width", str(int(width)), "--height", str(int(height)), "-q", str(quant)]
    extra = ["--output", str(out), "--guidance", str(float(guidance))]
    if (negative or "").strip():
        extra += ["--negative-prompt", negative.strip()]

    started = time.time()
    progress(0.1, desc="Generating… first run downloads ~33 GB, then this is faster")
    try:
        proc = _run(core + extra, OUT_DIR)
        if proc.returncode != 0 and any(h in (proc.stderr or "").lower() for h in _ARG_ERR_HINTS):
            # Older/different mflux: retry with just the confirmed core flags. mflux
            # writes into cwd (OUT_DIR); we pick up whatever PNG it produced.
            proc = _run(core, OUT_DIR)
    except subprocess.TimeoutExpired:
        return None, "Timed out after 60 minutes."
    except Exception as e:  # noqa: BLE001
        return None, f"Failed to launch mflux: {e}"

    log = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    img = out if out.exists() else _newest_png(started - 1, OUT_DIR, Path.cwd())
    if img and Path(img).exists():
        dt = time.time() - started
        return str(img), (f"✅ Saved {Path(img).name}\n   → {OUT_DIR}\n"
                          f"   {dt:.0f}s · seed {seed} · {width}×{height} · {steps} steps · q{quant}\n\n"
                          f"{log[-1200:]}")
    return None, f"No image produced (exit {proc.returncode}).\n\n{log[-3000:]}"


with gr.Blocks(title="Qwen-Image Studio") as demo:
    gr.Markdown(
        "# Qwen-Image Studio\n"
        "Local text-to-image on Apple Silicon (mflux / MLX). "
        "**Unload the big LLM in LM Studio first** — Qwen-Image and the 235B model "
        "can't both fit in memory. First generate downloads ~33 GB of weights."
    )
    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(label="Prompt", lines=3,
                                placeholder="A cinematic photograph of …")
            negative = gr.Textbox(label="Negative prompt (optional)", lines=1)
            with gr.Row():
                width = gr.Slider(512, 2048, 1024, step=64, label="Width")
                height = gr.Slider(512, 2048, 1024, step=64, label="Height")
            with gr.Row():
                steps = gr.Slider(10, 60, 40, step=1, label="Steps")
                guidance = gr.Slider(1.0, 7.0, 1.0, step=0.5,
                                     label="Guidance (1 = off, recommended)")
            with gr.Row():
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                quant = gr.Dropdown(["4", "6", "8"], value=DEFAULT_QUANT,
                                    label="Quantization (bits)")
            with gr.Row():
                go = gr.Button("Generate", variant="primary", scale=3)
                free = gr.Button("Free LLM memory", scale=1)
        with gr.Column(scale=1):
            out_img = gr.Image(label="Result", type="filepath")
            status = gr.Textbox(label="Status / log", lines=12)
    go.click(generate,
             [prompt, negative, width, height, steps, seed, quant, guidance],
             [out_img, status])
    free.click(free_llm, None, status)


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=PORT, inbrowser=False)

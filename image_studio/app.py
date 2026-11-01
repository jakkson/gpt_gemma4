#!/usr/bin/env python3
"""Qwen-Image Studio — standalone local text-to-image for the MacBook (MLX/mflux).

Deliberately SEPARATE from the RAG search app (photo_index/gradio_app.py):
  * Memory: a large image model can't coexist with the big LLM on a 128 GB Mac —
    unload the LLM (button below) before generating the heavy models.
  * Runtime: image gen runs through mflux/MLX (its own process + venv), not the
    LM Studio OpenAI API the RAG app uses.
  * It must NEVER run on the 32 GB mini — it would OOM/panic alongside the 30B.

Models (pick per-image):
  * Z-Image Turbo  — ~6 GB, fastest (~9 steps). Default.
  * FLUX.1-schnell — ~7 GB, fast (4 steps), better on complex/layered scenes.
  * Qwen-Image 2.1 — ~33 GB, slow (40 steps), best text-in-image + detail.
Each downloads its weights from Hugging Face on first use.

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
PORT = int(os.environ.get("IMAGE_STUDIO_PORT", "7863"))
DEFAULT_QUANT = os.environ.get("IMAGE_STUDIO_QUANT", "8")

# Image models via mflux. Each maps to its CLI command, optional --model arg, a
# recommended step count, and whether it honors guidance/negative (distilled
# "turbo/schnell" models run with CFG off, so those controls don't apply to them).
MODELS: dict[str, dict] = {
    "Z-Image Turbo — ~6 GB, fastest": {
        "cmd": "mflux-generate-z-image-turbo", "steps": 9, "guidance": False,
    },
    "FLUX.1-schnell — ~7 GB, fast": {
        "cmd": "mflux-generate", "model_arg": "schnell", "steps": 4, "guidance": False,
    },
    "Qwen-Image 2.1 — ~33 GB, best text/detail": {
        "cmd": "mflux-generate-qwen-2.1", "steps": 40, "guidance": True,
    },
}
def _default_model() -> str:
    env = os.environ.get("IMAGE_STUDIO_MODEL", "").strip().lower()
    if env:
        for k in MODELS:
            if env in k.lower():
                return k
    return next(iter(MODELS))


DEFAULT_MODEL = _default_model()

# LM Studio CLI, used by the "Free LLM memory" button to unload the big model.
LMS_CMD = os.environ.get("LMS_CMD", "lms")

# Flags dropped on an "unrecognized argument" retry, so the app survives mflux
# version/flag drift across models.
_ARG_ERR_HINTS = ("unrecognized arguments", "no such option", "unexpected extra",
                  "error: argument", "invalid choice", "usage:")


def _resolve_cmd(name: str) -> str | None:
    """mflux CLIs live next to this venv's python; PATH isn't set when launched via
    ./.venv/bin/python, so look there first, then fall back to PATH."""
    cand = Path(sys.executable).parent / name
    if cand.exists():
        return str(cand)
    return shutil.which(name)


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


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=3600)


def free_llm():
    """Unload ALL LM Studio models so their memory frees up for image generation.
    Model-agnostic on purpose — it doesn't matter which LLM is loaded."""
    exe = shutil.which(LMS_CMD)
    if not exe:
        return ("LM Studio CLI `lms` not on PATH. One-time: open LM Studio and run its "
                "`lms bootstrap`, or just eject the model manually in LM Studio before "
                "generating the heavy models.")
    try:
        p = subprocess.run([exe, "unload", "--all"], capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return f"Failed to run `lms unload --all`: {e}"
    msg = ((p.stdout or "") + "\n" + (p.stderr or "")).strip()
    if p.returncode == 0:
        return f"✅ Unloaded LM Studio models — memory freed for image gen.\n{msg}"
    return f"`lms unload --all` exit {p.returncode}:\n{msg}"


def on_model_change(model_key):
    """Set the step slider to the chosen model's recommended default."""
    spec = MODELS.get(model_key, {})
    return gr.update(value=spec.get("steps", 20))


def generate(model_key, prompt, negative, width, height, steps, seed, quant, guidance,
             progress=gr.Progress()):
    prompt = (prompt or "").strip()
    if not prompt:
        return None, "Enter a prompt first."
    spec = MODELS.get(model_key) or MODELS[DEFAULT_MODEL]
    exe = _resolve_cmd(spec["cmd"])
    if not exe:
        return None, (f"`{spec['cmd']}` not found in the venv. Run image_studio/setup.sh, "
                      "or check `ls image_studio/.venv/bin | grep mflux` — the command "
                      "name can vary by mflux version.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seed = int(seed) if seed is not None else -1
    if seed < 0:
        seed = int(time.time()) % 2_000_000_000
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = "".join(c for c in prompt[:40] if c.isalnum() or c in " -_").strip().replace(" ", "_")
    out = OUT_DIR / f"{ts}_{slug or 'image'}_seed{seed}.png"

    core = [exe]
    if spec.get("model_arg"):
        core += ["--model", spec["model_arg"]]
    core += ["--prompt", prompt, "--steps", str(int(steps)), "--seed", str(seed),
             "--width", str(int(width)), "--height", str(int(height)), "-q", str(quant)]
    extra = ["--output", str(out)]
    if spec.get("guidance"):  # only CFG models honor guidance / negative prompt
        extra += ["--guidance", str(float(guidance))]
        if (negative or "").strip():
            extra += ["--negative-prompt", negative.strip()]

    started = time.time()
    progress(0.1, desc=f"Generating with {model_key.split(' — ')[0]} — first use downloads weights")
    try:
        proc = _run(core + extra, OUT_DIR)
        if proc.returncode != 0 and any(h in (proc.stderr or "").lower() for h in _ARG_ERR_HINTS):
            proc = _run(core, OUT_DIR)  # retry with core flags only (version drift)
    except subprocess.TimeoutExpired:
        return None, "Timed out after 60 minutes."
    except Exception as e:  # noqa: BLE001
        return None, f"Failed to launch mflux: {e}"

    log = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    img = out if out.exists() else _newest_png(started - 1, OUT_DIR, Path.cwd())
    if img and Path(img).exists():
        dt = time.time() - started
        return str(img), (f"✅ Saved {Path(img).name}\n   → {OUT_DIR}\n"
                          f"   {dt:.0f}s · {model_key.split(' — ')[0]} · seed {seed} · "
                          f"{width}×{height} · {steps} steps · q{quant}\n\n{log[-1200:]}")
    return None, f"No image produced (exit {proc.returncode}).\n\n{log[-3000:]}"


with gr.Blocks(title="Image Studio") as demo:
    gr.Markdown(
        "# Image Studio\n"
        "Local text-to-image on Apple Silicon (mflux / MLX). Pick a model: "
        "**Z-Image Turbo** (small/fast), **FLUX.1-schnell** (fast, complex scenes), "
        "or **Qwen-Image 2.1** (big, best text/detail). Heavy models: **Free LLM memory** "
        "first. First use of each model downloads its weights."
    )
    with gr.Row():
        with gr.Column(scale=1):
            model = gr.Dropdown(list(MODELS), value=DEFAULT_MODEL, label="Model")
            prompt = gr.Textbox(label="Prompt", lines=3,
                                placeholder="A cinematic photograph of …")
            negative = gr.Textbox(label="Negative prompt (Qwen only)", lines=1)
            with gr.Row():
                width = gr.Slider(512, 2048, 1024, step=64, label="Width")
                height = gr.Slider(512, 2048, 1024, step=64, label="Height")
            with gr.Row():
                steps = gr.Slider(1, 60, MODELS[DEFAULT_MODEL]["steps"], step=1, label="Steps")
                guidance = gr.Slider(1.0, 7.0, 1.0, step=0.5,
                                     label="Guidance (Qwen; 1 = off)")
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

    model.change(on_model_change, model, steps)
    go.click(generate,
             [model, prompt, negative, width, height, steps, seed, quant, guidance],
             [out_img, status])
    free.click(free_llm, None, status)


if __name__ == "__main__":
    # The gradio_client "'const' in schema" crash is fixed by the modern gradio
    # (requirements pin >=5 on Python 3.11+); no launch-time workaround needed.
    demo.queue().launch(server_name="127.0.0.1", server_port=PORT, inbrowser=False)

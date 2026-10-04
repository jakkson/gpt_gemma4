# Qwen-Image Studio

Standalone local **text-to-image** generator for the MacBook (Apple Silicon),
using **Qwen-Image 2.1** via **mflux / MLX**. Intentionally separate from the RAG
search app — see why below.

## Why it's its own app (not a tab in the search UI)
- **Memory:** Qwen-Image needs ~20 GB resident; it cannot coexist with the
  ~103 GB Qwen3-235B LLM on a 128 GB Mac. **Unload the LLM in LM Studio before
  generating.** A separate app makes that load/unload lifecycle natural.
- **Runtime:** image gen runs through mflux/MLX (its own process + venv), not the
  LM Studio OpenAI API the search app uses. LM Studio can't host diffusion models.
- **Never on the mini:** the 32 GB mini runs the shared search app + the 30B;
  adding a diffusion model there would OOM / kernel-panic. This tool is MacBook-only.

## Setup (once, on the MacBook)
```bash
cd ~/gpt-local-gemma/image_studio
./setup.sh
```
Creates `image_studio/.venv`, installs `mflux` + `gradio`, and makes
`~/Generated Images`.

## Run
```bash
cd ~/gpt-local-gemma/image_studio
./.venv/bin/python app.py
# open http://127.0.0.1:7863
```
Or copy **`Image Studio.command`** to `~/Launchers` (or the Desktop) and
double-click it.

- **First generate downloads ~33 GB** of Qwen-Image weights from Hugging Face,
  then quantizes to the chosen bits (default **q8**). Later runs reuse the cache.
- Output PNGs are saved to **`~/Generated Images`** (override with
  `IMAGE_STUDIO_OUT`). Each file is named `<timestamp>_<prompt-slug>_seed<N>.png`.
- Recommended settings: **40 steps, guidance 1.0 (off)**. Seed `-1` = random.
- **"Free LLM memory"** button runs `lms unload --all` to eject the big LLM from
  LM Studio so its memory is available for image gen (needs the `lms` CLI on PATH;
  otherwise eject manually in LM Studio first).

## Loading the 235B LLM at all (separate from this app)
If LM Studio refuses to load a ~100 GB model despite free RAM, raise the macOS GPU
wired-memory cap (default ~75% of RAM blocks big models) and set LM Studio's
**Model loading guardrails → Off**:
```bash
sudo sysctl iogpu.wired_limit_mb=122880   # 120 GB; resets on reboot
```

## Overrides (env vars)
| Var | Default | Purpose |
|---|---|---|
| `MFLUX_QWEN_CMD` | `mflux-generate-qwen-2.1` | mflux command (set to `mflux-generate-qwen` for the base model, or whatever your mflux version installs) |
| `IMAGE_STUDIO_OUT` | `~/Generated Images` | where PNGs are saved |
| `IMAGE_STUDIO_PORT` | `7863` | web UI port |
| `IMAGE_STUDIO_QUANT` | `8` | default quantization bits |

If the app says the command isn't found, check `ls .venv/bin | grep mflux` and set
`MFLUX_QWEN_CMD` to the name you see.

# autoresearch-lora: Design Spec

Autonomous LoRA training loop for image generation on Apple Silicon. An LLM proposes experiments by editing a config file, a fixed pipeline trains LoRAs via mflux, generates eval images, and scores them with CLIP. The LLM keeps or discards based on results and loops indefinitely.

Adapted from [autoresearch-mlx](https://github.com/TheGermanAZ/autoresearch-mlx) — same loop pattern, different domain.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   LLM (Claude)                       │
│                                                      │
│  Reads: results.tsv, reasoning.md, config.yaml       │
│  Writes: config.yaml (new experiment)                │
│          reasoning.md (why this experiment)           │
│          results.tsv (after scoring)                  │
└──────────────────────┬──────────────────────────────┘
                       │ edits config.yaml
                       ▼
┌─────────────────────────────────────────────────────┐
│              config.yaml (ONLY mutable file)          │
│                                                      │
│  rank, alpha, lr, batch_size, steps, num_epochs,     │
│  quantize (4/6/8-bit), guidance, target_layers,      │
│  trigger_word, caption_template                      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│      train.py — Subprocess Orchestrator (fixed)      │
│                                                      │
│  1. Translate config.yaml → mflux JSON config        │
│  2. subprocess: mflux-train (timeout=TIME_BUDGET)    │
│  3. subprocess: mflux-generate ×24 (timeout each)    │
│  4. Compute CLIP image-image similarity              │
│  5. Print summary                                    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  Output (run.log)                                    │
│                                                      │
│  clip_sim_centroid, clip_sim_nn, prompt_scores,      │
│  score_stddev, neg_control, peak_vram_mb,            │
│  training_seconds, steps_completed, eval_seconds     │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **config.yaml as editable surface** — LLM tunes structured data, not code. Avoids syntax errors. YAML is easier for LLMs to edit reliably than Python.
- **train.py is a subprocess orchestrator** — shells out to `mflux-train` and `mflux-generate`, not a Python training loop. Different from the original autoresearch where train.py contains the model.
- **Separate repo** — not a modification of autoresearch-mlx.
- **mflux handles the heavy lifting** — model loading, LoRA training, image generation all via mflux CLI.
- **reasoning.md journal** — LLM writes what it's trying and why. Fascinating to read, helps it make smarter choices with fewer iterations.

### File Layout

```
autoresearch-lora/
├── program.md            # Instructions for the LLM
├── config.yaml           # Mutable: hyperparams + captions
├── train.py              # Fixed: subprocess orchestrator
├── prepare.py            # One-time: download model, compute ref centroid
├── eval_prompts.txt      # 6 fixed eval prompts
├── results.tsv           # Experiment log
├── reasoning.md          # LLM's research journal
├── reference_images/     # User's training images (5-20 photos)
└── eval_images/          # Latest 24 generated eval images
```

### Constants (fixed in train.py)

| Constant | Value | Rationale |
|----------|-------|-----------|
| TIME_BUDGET | 900s (15 min) | LoRA training is slower per step than LLM pretraining. 15 min gives enough steps for signal. |
| TOTAL_TIMEOUT | 1500s (25 min) | Training + eval + scoring. Hard cap via `timeout 1800`. |
| EVAL_RESOLUTION | 1024×1024 | Klein 4B's native training resolution. 512 produces degraded output. |
| EVAL_SEEDS | [42, 137, 256, 999] | Fixed seeds for deterministic comparison across experiments. |
| EVAL_STEPS | 20 | Inference steps per image. Calibrated during smoke test. |
| BASE_MODEL | FLUX.2 Klein 4B | Smallest Flux model. Fast training, fits in 36GB. |

### Search Space (config.yaml)

| Parameter | Type | Description |
|-----------|------|-------------|
| rank | int | LoRA rank (4–128) |
| alpha | int | LoRA alpha scaling |
| lr | float | Learning rate |
| batch_size | int | Training batch size |
| steps | int | Training iterations |
| num_epochs | int | Number of epochs |
| quantize | int | Base model quantization (4/6/8-bit) |
| guidance | float | Classifier-free guidance scale |
| target_layers | string | Which layers get adapters (locked to "default" initially) |
| trigger_word | string | Token that activates the LoRA |
| caption_template | string | Template for training captions |

**Note:** `target_layers` is locked to mflux's default attention layers initially. The combinatorial space is enormous — only unlock after exhausting simpler hyperparameter tuning.

### Memory Budget

- Klein 4B base model: ~8GB (bf16)
- LoRA training overhead: ~4-6GB
- Eval inference: ~2-4GB
- **Peak: ~16-20GB** on 36GB Mac. Comfortable margin.

---

## Eval Prompts & Scoring

### Eval Prompts (eval_prompts.txt)

6 fixed prompts with `{trigger}` placeholder:

1. **Identity:** `a photo of {trigger}`
2. **Context:** `a photo of {trigger} in a coffee shop`
3. **Style:** `a watercolor painting of {trigger}`
4. **Action:** `{trigger} laughing, candid photo`
5. **Composition:** `a close-up portrait of {trigger}, soft lighting`
6. **Negative:** `a photo of a landscape at sunset` (no trigger — control)

Prompts 1-5 test identity, context transfer, style transfer, action/pose, and composition. Prompt 6 is a negative control that catches overfitting/leaking. User customizes these during setup. Structure stays fixed.

### Primary Metric

**`clip_sim_centroid`** is the single keep/discard criterion. Higher is better. An experiment is kept if it improves by ≥ 0.005 over the previous best.

All other metrics (clip_sim_nn, per-prompt scores, stddev, neg_control) are secondary signals the LLM uses for reasoning but do not drive the keep/discard decision.

### Scoring Pipeline

**PREPARE (one-time, in prepare.py):**
1. Load CLIP model (mlx_clip, MLX-native)
2. Compute CLIP embedding for each reference image
3. Compute centroid = mean(all ref embeddings)
4. Save to `~/.cache/autoresearch-lora/`:
   - `ref_centroid.npy`
   - `ref_embeddings.npy` (for nearest-neighbor)
5. Run negative control baseline, record score

**EVAL (every experiment, in train.py):**
1. Generate 24 images: 6 prompts × 4 seeds, 1024×1024, 20 inference steps, sequential
2. Score prompts 1-5 (20 images): CLIP embedding → cosine sim vs centroid + nearest ref
3. Score prompt 6 (4 images): CLIP embedding → cosine sim vs centroid (separate)
4. Aggregate:
   - `clip_sim_centroid` = mean(20 centroid sims, prompts 1-5)
   - `clip_sim_nn` = mean(20 nearest-neighbor sims)
   - `prompt_scores` = [mean(4 seeds) for each prompt 1-5]
   - `score_stddev` = std(20 centroid sims)
   - `neg_control` = mean(4 seeds, prompt 6)
5. If `neg_control > 0.45` → warn (informational only, does not block keep/discard)

### Output Format

```
---
clip_sim_centroid:  0.847
clip_sim_nn:        0.812
prompt_scores:      0.85, 0.83, 0.87, 0.86, 0.84
score_stddev:       0.018
neg_control:        0.312
peak_vram_mb:       16400
training_seconds:   900.0
steps_completed:    680
eval_seconds:       112.4
---
```

### results.tsv Schema

```
commit	clip_centroid	clip_nn	stddev	neg_ctrl	memory_gb	status	description
a1b2c3d	0.847	0.812	0.018	0.312	16.0	keep	baseline (default config)
e4f5g6h	0.862	0.831	0.015	0.305	16.0	keep	rank 8 → 16
i7j8k9l	0.859	0.828	0.022	0.318	14.2	discard	quantize 8 → 4 (within noise)
```

---

## Experiment Loop

### Setup (one-time)

1. Create branch: `autoresearch-lora/<tag>`
2. Verify prepare.py has been run (model downloaded, ref centroid computed)
3. Initialize results.tsv with header
4. Run baseline with default config: `timeout 1800 uv run train.py > run.log 2>&1`
5. Record baseline in results.tsv + reasoning.md

### Loop

```
LOOP FOREVER:

1. READ STATE
   • results.tsv (all past experiments)
   • reasoning.md (recent entries — last ~10)
   • config.yaml (current config)

2. REASON
   • What has worked? What hasn't?
   • What is the highest-leverage thing to try?
   • Write hypothesis in reasoning.md (append, keep entries short)

3. EDIT config.yaml
   • Change ONE thing (or a deliberate combo)
   • Stage + commit:
     git add autoresearch-lora/config.yaml autoresearch-lora/reasoning.md
     git commit -m "experiment: <description>"

4. RUN
   timeout 1800 uv run train.py > run.log 2>&1

5. READ RESULTS
   grep "^clip_sim_centroid:" run.log
   If empty → tail -50 run.log (crash or timeout)

6. DECIDE
   Higher clip_sim_centroid = better. Min delta: 0.005.

   If improved (delta >= 0.005):
     → Log to results.tsv (status: keep)
     → git add autoresearch-lora/results.tsv
     → git commit -m "results: keep <description>"

   If worse or within noise (delta < 0.005):
     → Log to results.tsv (status: discard)
     → Revert config: git checkout HEAD~1 -- autoresearch-lora/config.yaml
     → git add autoresearch-lora/config.yaml autoresearch-lora/results.tsv
     → git commit -m "revert: <description>"

   If crash or timeout:
     → Log to results.tsv (status: crash)
     → Revert config: git checkout HEAD~1 -- autoresearch-lora/config.yaml
     → git add + commit as above
     → Diagnose. Trivial fix → retry. Fundamental → skip.

7. GOTO 1
```

### Crash Recovery

If starting a new session, read results.tsv and git log to determine last completed experiment. Resume from step 1.

### NEVER STOP

Once the loop begins, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be asleep or away. You are autonomous. If you run out of ideas, re-read reasoning.md, try combining near-misses, try more radical changes. The loop runs until the human interrupts you.

### Strategy Guidance (suggestions, not rules)

- **Phase 1:** Find the best rank. Start with 8, try neighbors based on results. Follow the gradient, don't sweep.
- **Phase 2:** Given best rank, explore learning rate. Try 2x and 0.5x, narrow in.
- **Phase 3:** Explore quantize (4, 6, 8-bit). Trades model fidelity for more training steps.
- **Phase 4:** Caption strategies. Trigger words, detail level, templates. Often the biggest lever.
- **Phase 5:** Combinations + refinement around best known config.

Adapt based on results. If rank barely matters but LR is sensitive, spend more time on LR.

### Cadence

~25 min per experiment. ~2.4 experiments/hour. **~19 experiments overnight** (8 hours).

---

## Setup Flow (prepare.py)

### Usage

```bash
uv run prepare.py --images ./my-photos/
```

### Storage

All cached data lives in `~/.cache/autoresearch-lora/`:

```
~/.cache/autoresearch-lora/
├── images/              # Validated copies of reference images
├── ref_centroid.npy     # Mean CLIP embedding of references
├── ref_embeddings.npy   # Individual CLIP embeddings (for NN)
└── neg_baseline.txt     # Baseline negative control score
```

### Steps

1. **Download model** — Pre-download FLUX.2 Klein 4B via `mflux-save` (~8-10GB). Avoids blowing the first experiment's timeout.

2. **Validate + copy images** — Check source directory: 5-20 images, .jpg/.jpeg/.png, min 512×512. Copy to `~/.cache/autoresearch-lora/images/`.

3. **Compute reference embeddings** — Load mlx_clip. Embed each reference image. Compute centroid (mean). Save centroid + individual embeddings to `.npy` files.

4. **Initialize config** — Write default `config.yaml`:
   ```yaml
   rank: 8
   alpha: 16
   lr: 3e-4
   batch_size: 1
   steps: 1000
   num_epochs: 1
   quantize: 4
   guidance: 4.0
   target_layers: "default"
   trigger_word: "ohwx"
   caption_template: "a photo of {trigger}"
   ```

5. **Write eval prompts** — Generate `eval_prompts.txt` (6 prompts). User confirms or edits.

6. **Smoke test** — Validates full pipeline:
   - (a) 5 LoRA training steps, 1 image, batch_size=1 → pass: no crash
   - (b) Generate 1 image, 1024×1024, seed=42, 20 steps → pass: valid PNG
   - (c) CLIP score generated image vs centroid → pass: similarity > 0.0
   - (d) Negative baseline: generate 1 landscape (no trigger), record score
   - (e) Inference step calibration: time 1 image at 8, 12, 20 steps. Print timing so user can verify 20 steps is feasible (24 images × T seconds = eval time)

### Dependencies

```toml
[project]
name = "autoresearch-lora"
requires-python = ">=3.11"
dependencies = [
    "mflux>=0.16.0,<0.17",
    "mlx>=0.30.0,<0.32",
    "mlx-clip",
    "numpy",
    "pillow",
    "pyyaml>=6.0",
]
```

---

## Pre-Implementation Validations

Before writing code, these 3 things must be validated hands-on:

1. **mflux CLI end-to-end** — Install mflux, train a dummy LoRA on Klein 4B, generate an image with it. Record exact commands, JSON config schema, and output paths. Pin to the exact mflux version tested.

2. **CLIP on MLX** — Install mlx_clip, embed a reference image and a generated image, compute cosine similarity. Confirm it runs without PyTorch.

3. **Pin mflux version** — API has changed significantly across versions. Build against the exact config schema of the tested version.

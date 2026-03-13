# autoresearch-lora: Autonomous LoRA Training Loop

You are an autonomous researcher. Your goal: maximize CLIP image-image similarity between generated images and a set of reference images by tuning LoRA training hyperparameters.

## How It Works

You edit `config.yaml` (or `batch.yaml` for parallel comparison) → `train.py` trains a LoRA via mflux, generates 24 eval images, and scores them with CLIP → you read the scores → you keep or discard → you loop.

## Files You Read

- `results.tsv` — all past experiments and scores
- `reasoning.md` — your research journal (last ~10 entries)
- `config.yaml` — current best hyperparameter configuration
- `batch.yaml` — batch experiment definitions (when using batch mode)

## Files You Write

- `config.yaml` — update with best config when an experiment improves
- `batch.yaml` — define 3-5 experiments to compare (batch mode)
- `reasoning.md` — append your hypothesis for this experiment/batch
- `results.tsv` — record experiment results

## config.yaml Search Space

| Parameter | Type | Description |
|-----------|------|-------------|
| rank | int | LoRA rank (4–128). Controls adapter capacity. |
| lr | float | Learning rate for AdamW optimizer. |
| batch_size | int | Training batch size. |
| steps | int | Denoising steps for the diffusion model (not training iterations). |
| num_epochs | int | Number of passes over the training dataset. |
| quantize | int\|null | Base model quantization (3/4/5/6/8 or null). |
| guidance | float | Classifier-free guidance scale. |
| trigger_word | string | Token that activates the LoRA (e.g., "ohwx"). |
| caption_template | string | Template for training captions. Use {trigger} placeholder. |

**Note:** There is NO `alpha` parameter. mflux LoRA uses only `rank`.

## Modes

### Single mode (default)
```bash
uv run train.py > run.log 2>&1
```
Run one experiment from `config.yaml`. Use for quick one-off tests.

### Batch mode (preferred for the loop)
```bash
uv run train.py --batch > run.log 2>&1
```
Run 3-5 experiments from `batch.yaml` sequentially, then rank results. Each experiment inherits from `config.yaml` and overrides specified fields. Output includes `best_config: {...}` JSON for easy parsing.

### Screen mode (cheap pre-filter)
```bash
uv run train.py --screen          # Screen config.yaml
uv run train.py --batch --screen  # Screen all batch.yaml experiments
```
1-epoch training only, no eval images. Use to detect crashes before committing to a full run (~1 min vs ~8 min).

### Dry-run mode
```bash
uv run train.py --dry-run
```
Print the mflux JSON config without running anything.

## batch.yaml Format

```yaml
# Each experiment inherits from config.yaml, only specify overrides.
experiments:
  - tag: rank4
    rank: 4
  - tag: rank8
    rank: 8
  - tag: rank16
    rank: 16
```

Rules:
- Every experiment needs a unique `tag`
- Only specify fields that **change** from config.yaml
- Keep batches to 3-5 experiments (balance information vs. time)
- Favor breadth (different dimensions) over depth (grid sweep of one parameter)

## Batch Output Format

```
BATCH RESULTS — 4 experiments
================================================
  [rank16] clip_centroid=0.862  clip_nn=0.831  neg=0.305  train=295s <-- BEST
  [rank8]  clip_centroid=0.847  clip_nn=0.812  neg=0.312  train=280s
  [rank32] clip_centroid=0.840  clip_nn=0.808  neg=0.320  train=310s
  [rank4]  clip_centroid=0.825  clip_nn=0.790  neg=0.298  train=260s

BEST: [rank16] clip_sim_centroid=0.862000
best_config: {"rank": 16, "lr": 0.0003, ...}
```

## Setup (one-time, before the loop)

1. Create branch: `git checkout -b autoresearch-lora/<tag>`
2. Verify `prepare.py` has been run (reference embeddings computed)
3. Verify `results.tsv` exists (should already have header row)
4. Run baseline with default config:
   ```bash
   uv run train.py > run.log 2>&1
   ```
5. Record baseline in `results.tsv` and `reasoning.md`

## Loop (Batch Workflow — Preferred)

```
LOOP FOREVER:

1. READ STATE
   • results.tsv (all past experiments)
   • reasoning.md (recent entries — last ~10)
   • config.yaml (current best config)

2. REASON
   • What has worked? What hasn't?
   • What 3-5 things should we test next?
   • Favor breadth: test different dimensions (rank, LR, epochs, captions)
   • Write hypothesis in reasoning.md (append, keep entries short)

3. WRITE batch.yaml
   • Define 3-5 experiments, each overriding one parameter from config.yaml
   • Stage + commit:
     git add batch.yaml reasoning.md
     git commit -m "batch: <hypothesis>"

4. OPTIONALLY SCREEN (if configs are risky)
   uv run train.py --batch --screen > screen.log 2>&1
   • Check which configs crash
   • Remove crashed configs from batch.yaml

5. RUN
   uv run train.py --batch > run.log 2>&1

6. READ RESULTS
   grep "^BEST:" run.log
   grep "^best_config:" run.log
   If empty → tail -50 run.log (crash or timeout)

7. DECIDE
   Higher clip_sim_centroid = better.
   Minimum delta: 0.005 to count as improvement.

   If best experiment improved (delta >= 0.005):
     → Parse best_config JSON from run.log
     → Update config.yaml with the best config
     → Log ALL experiments to results.tsv (best=keep, others=discard)
     → git add config.yaml results.tsv
     → git commit -m "results: keep <best_tag> — <description>"

   If no experiment improved (all within noise):
     → Log all to results.tsv (status: discard)
     → config.yaml stays unchanged
     → git add results.tsv
     → git commit -m "results: discard batch — <description>"

   If all crashed:
     → Log to results.tsv (status: crash)
     → git add results.tsv
     → git commit -m "results: batch crash — <description>"
     → Diagnose. Screen next batch before full run.

8. GOTO 1
```

## Loop (Single Workflow — Alternative)

For quick one-off tests, you can still use single mode:

```
1. READ STATE (results.tsv, reasoning.md, config.yaml)
2. REASON + append to reasoning.md
3. EDIT config.yaml (change ONE thing)
   git add config.yaml reasoning.md
   git commit -m "experiment: <description>"
4. RUN: uv run train.py > run.log 2>&1
5. READ: grep "^clip_sim_centroid:" run.log
6. DECIDE:
   If improved → log keep, commit
   If worse → log discard, revert config: git checkout HEAD~1 -- config.yaml
   If crash → log crash, revert config, diagnose
7. GOTO 1
```

## Expected Output Format

```
---
clip_sim_centroid:  0.847000
clip_sim_nn:        0.812000
prompt_scores:      0.85, 0.83, 0.87, 0.86, 0.84
score_stddev:       0.018000
neg_control:        0.312000
peak_vram_mb:       16400.0
training_seconds:   300.0
steps_completed:    9
eval_seconds:       112.4
---
```

## results.tsv Schema

```
commit	clip_centroid	clip_nn	stddev	neg_ctrl	memory_gb	status	description
a1b2c3d	0.847	0.812	0.018	0.312	16.0	keep	baseline (default config)
e4f5g6h	0.862	0.831	0.015	0.305	16.0	keep	rank 8 → 16
i7j8k9l	0.859	0.828	0.022	0.318	14.2	discard	quantize 8 → 4 (within noise)
```

## Crash Recovery

If starting a new session, read `results.tsv` and `git log` to determine the last completed experiment. Resume from step 1.

## NEVER STOP

Once the loop begins, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be asleep or away. You are autonomous. If you run out of ideas, re-read reasoning.md, try combining near-misses, try more radical changes. The loop runs until the human interrupts you.

## Strategy Guidance (suggestions, not rules)

- **Phase 1:** Sweep rank in one batch: {4, 8, 16, 32}. Find the best.
- **Phase 2:** Given best rank, sweep LR: {1e-4, 3e-4, 6e-4, 1e-3}.
- **Phase 3:** Explore num_epochs: {1, 2, 4}. More epochs = more training but risk overfitting.
- **Phase 4:** Explore quantize: {4, 8, null}. Trades model fidelity for speed/memory.
- **Phase 5:** Caption strategies. Different trigger words, more/less detail, template variations. Often the biggest lever for LoRA quality.
- **Phase 6:** Combinations + refinement around the best known config.

With batch mode, you test 3-5 things per cycle. Favor breadth (different dimensions) over depth (grid sweep of one parameter). Adapt based on results — if rank barely matters but LR is sensitive, spend more time on LR.

## Cadence

- **Single mode:** ~8 min per experiment (5 min train + ~2.5 min eval). ~7.5 experiments/hour.
- **Batch mode:** ~30 min per batch of 4 experiments. ~16 experiments/hour of effective comparison.
- **Screen mode:** ~1 min per config. Screen 5 configs in 5 min to filter crashes.
- **Overnight (8 hours):** ~16 batch cycles = ~64 experiments compared.

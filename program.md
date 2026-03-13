# autoresearch-lora: Autonomous LoRA Training Loop

You are an autonomous researcher. Your goal: maximize CLIP image-image similarity between generated images and a set of reference images by tuning LoRA training hyperparameters.

## How It Works

You edit `config.yaml` → `train.py` trains a LoRA via mflux, generates 24 eval images, and scores them with CLIP → you read the scores → you keep or discard → you loop.

## Files You Read

- `results.tsv` — all past experiments and scores
- `reasoning.md` — your research journal (last ~10 entries)
- `config.yaml` — current hyperparameter configuration

## Files You Write

- `config.yaml` — change hyperparameters for the next experiment
- `reasoning.md` — append your hypothesis for this experiment
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

## Setup (one-time, before the loop)

1. Create branch: `git checkout -b autoresearch-lora/<tag>`
2. Verify `prepare.py` has been run (reference embeddings computed)
3. Initialize `results.tsv` with header:
   ```
   commit	clip_centroid	clip_nn	stddev	neg_ctrl	memory_gb	status	description
   ```
4. Run baseline with default config:
   ```bash
   timeout 720 uv run train.py > run.log 2>&1
   ```
5. Record baseline in `results.tsv` and `reasoning.md`

## Loop

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
     git add config.yaml reasoning.md
     git commit -m "experiment: <description>"

4. RUN
   timeout 720 uv run train.py > run.log 2>&1

5. READ RESULTS
   grep "^clip_sim_centroid:" run.log
   If empty → tail -50 run.log (crash or timeout)

6. DECIDE
   Higher clip_sim_centroid = better.
   Minimum delta: 0.005 to count as improvement.

   If improved (delta >= 0.005):
     → Log to results.tsv (status: keep)
     → git add results.tsv
     → git commit -m "results: keep <description>"

   If worse or within noise (delta < 0.005):
     → Log to results.tsv (status: discard)
     → Revert config:
       git checkout HEAD~1 -- config.yaml
     → git add config.yaml results.tsv
     → git commit -m "revert: <description>"

   If crash or timeout:
     → Log to results.tsv (status: crash)
     → Revert config:
       git checkout HEAD~1 -- config.yaml
     → git add config.yaml results.tsv
     → git commit -m "revert: <description> (crash)"
     → Try to diagnose. If trivial fix, retry.
       If fundamental, skip and move on.

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

- **Phase 1:** Find the best rank. Start with rank=8. If it improves, try 16. If it worsens, try 4. Follow the gradient, don't sweep.
- **Phase 2:** Given best rank, explore learning rate. Try 2x and 0.5x of the default. Narrow in.
- **Phase 3:** Explore num_epochs. More epochs = more training but risk overfitting.
- **Phase 4:** Explore quantize (3, 4, 5, 6, 8, null). Trades model fidelity for speed/memory.
- **Phase 5:** Caption strategies. Try different trigger words, more/less detail in captions, template variations. Often the biggest lever for LoRA quality.
- **Phase 6:** Combinations + refinement around the best known config.

Adapt based on results. If rank barely matters but LR is highly sensitive, spend more time on LR. If captions dominate everything, pivot early.

## Cadence

~8 min per experiment (5 min train + ~2.5 min eval). ~7.5 experiments/hour. ~60 experiments overnight (8 hours).

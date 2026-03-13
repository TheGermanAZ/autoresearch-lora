# Brainstorming Log: autoresearch-lora

Session date: 2026-03-12 → 2026-03-13

## Starting Point

We had just finished a session on [autoresearch-mlx](https://github.com/TheGermanAZ/autoresearch-mlx) — an autonomous LLM pretraining experiment loop on Apple Silicon. The branch `autoresearch/mar12c` achieved val_bpb 1.453 (down from 1.865 baseline) by replacing squared-ReLU MLP with SwiGLU and tuning hyperparameters. We merged that PR and discussed what other domains the autoresearch pattern could apply to.

## Domain Exploration

**Question:** What domains can an autoresearcher apply to without a ton of work?

The core pattern — LLM proposes change → runs experiment → measures single metric → keeps/discards → loops — transfers to anything with:
1. A single quantitative metric
2. A fast feedback loop (minutes, not hours)
3. A small editable surface
4. Deterministic-ish evaluation

**Candidates identified:**
- LoRA / fine-tuning recipes (same MLX stack)
- Prompt engineering (string editing + API eval)
- Compiler/build flag optimization (benchmark as metric)
- SQL query rewrites (execution time)
- Inference optimization (tokens/sec)

**Question:** Which has visual wow factor?

**Answer:** LoRA training for image generation. Every experiment produces images you can look at. The improvement arc is dramatic. "I trained an AI on 10 photos of my face overnight" is inherently shareable.

## Key Decisions

### Base Model
- **Decision:** FLUX.2 Klein 4B
- **Why:** Smallest Flux model, fastest training, mflux supports LoRA training for it natively on MLX. State of the art for its size class.
- **Alternatives considered:** Klein 9B (slower), Z-Image 6B (smaller ecosystem), configurable (added complexity)

### Evaluation Metric
- **Decision:** Hybrid CLIP + LLM judge → later simplified to CLIP image-image similarity only
- **Evolution:** Started with text-image CLIP, reviewer caught that this measures "is this a photo of a dog" not "is this MY dog." Changed to image-image similarity against reference centroid. LLM judge dropped for v1 (non-deterministic, expensive).
- **Final approach:** CLIP image-image cosine similarity vs reference centroid (primary) + nearest-neighbor (secondary)

### Search Space
- **Decision:** LoRA params + target layers + captions
- **What's in:** rank, alpha, LR, batch_size, steps, num_epochs, quantize, guidance, target_layers, trigger_word, caption_template
- **What's locked initially:** target_layers (too combinatorial, unlock later)

### Architecture Approach
- **Decision:** Hybrid — config file + LLM reasoning log
- **Why:** LLM edits config.yaml (structured data, not code). Also maintains reasoning.md journal explaining what it's trying and why. Results tracked in results.tsv. Same git keep/revert loop as original.
- **Alternatives rejected:** Direct code editing (unnecessary for LoRA), structured grid search (too rigid)

### Time Budget
- **Evolution:** Started at 15 min (reviewer concern about step count) → reduced to 5 min (matching original, validate via smoke test)
- **Rationale:** 5 min gives ~60 experiments overnight vs ~19 at 15 min. If scores are too noisy, bump up.

## Review Iterations

### Architecture (3 rounds)

**v1 → v2 fixes:**
- Resolution: 512 → 1024 (Klein 4B trained at 1024, 512 produces garbage)
- Time budget: 5 → 15 min (reviewer concern, later reverted)
- train.py is a subprocess orchestrator, not a training loop
- Added config knobs: steps, quantize, guidance
- Scoring: centroid-only → centroid + nearest-neighbor
- Added subprocess timeouts

**v2 → v3 fixes:**
- Updated architecture diagram with all changes
- Fixed cross-references (prompt count 5→6, image count 20→24, eval steps 8→20)

**v3:** Greenlit.

### Experiment Loop (2 rounds)

**v1 → v2 fixes:**
- `git commit --amend` → separate commit for results (crash-safe)
- `git reset --hard` → revert config + commit (preserves discard history)
- Added `timeout 1800` command (later `timeout 720`)
- Added NEVER STOP instruction
- Added metric direction: higher clip_sim = better, min delta 0.005
- Strategy phases: rigid sweep → direction-based ("follow the gradient")
- Added monorepo staging paths
- Added TSV header spec
- Added output format documentation
- Added crash recovery protocol
- Made revert command explicit: `git checkout HEAD~1 -- autoresearch-lora/config.yaml`

**v2:** Greenlit.

### Eval & Scoring + Setup Flow (2 rounds)

**v1 → v2 fixes:**
- Added primary metric declaration (clip_sim_centroid drives keep/discard)
- Added action/pose eval prompt (prompt 4)
- Eval images: 20 → 24 (6 prompts × 4 seeds)
- Inference steps: 8 → 20 (calibrated via smoke test)
- Neg control threshold: 0.8× centroid → absolute 0.45, informational only
- Added pyyaml to dependencies
- Tightened mlx pin: >=0.22 → >=0.30.0,<0.32
- Added storage path conventions (~/.cache/autoresearch-lora/)
- Smoke test: vague → 5 explicit checks + inference step calibration
- Added negative baseline recording during smoke test

**v2:** Greenlit.

### Consolidated Spec (2 rounds)

**Round 1 fixes:**
- Added LoRA artifact lifecycle section (where adapters are written, how they're referenced, cleanup)
- Added steps vs TIME_BUDGET semantics (mflux has no wall-clock cutoff, steps is the direct knob)
- Added timeout dependency note
- Clarified EVAL_STEPS mutability

**Round 2:** Approved.

### Final user change
- TIME_BUDGET: 15 min → 5 min (matching original, ~60 experiments overnight)

## Research Findings

### MLX Diffusion Landscape (as of March 2026)

| Library | Backend | Flux LoRA Train | Notes |
|---------|---------|----------------|-------|
| **mflux** | MLX | FLUX.2 Klein, Z-Image | Best path. Pure MLX, actively maintained. |
| mlx-examples | MLX | No | Inference only (SD 2.1, SDXL Turbo) |
| DiffusionKit | MLX/CoreML | No | Inference only |
| SimpleTuner | PyTorch/MPS | SDXL only | Flux blocked on Apple GPUs |
| Draw Things | Metal | Yes (GUI) | Not scriptable |

**Conclusion:** mflux is the only viable path for LoRA training on Apple Silicon with Flux models.

### Hardware Context
- Mac with 36GB unified memory
- Klein 4B at ~8GB + training overhead ~4-6GB = ~14-20GB peak
- Comfortable margin

## Deliverables

1. **Design spec:** `docs/design.md` — approved after 8 review rounds
2. **Implementation plan:** `docs/plans/2026-03-13-autoresearch-lora.md` — 7 tasks, 7 chunks
3. **New repo:** https://github.com/TheGermanAZ/autoresearch-lora
4. **Visual companion mockups:** `.superpowers/brainstorm/` in autoresearch-mlx repo

## Pre-Implementation Blockers

Three things must be validated hands-on before writing code:
1. mflux CLI end-to-end (train dummy LoRA, generate with it)
2. mlx_clip works for image-image similarity without PyTorch
3. Pin exact mflux version and record JSON config schema

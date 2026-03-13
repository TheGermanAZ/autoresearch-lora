# autoresearch-lora

Autonomous LoRA training loop for image generation on Apple Silicon. An LLM agent proposes hyperparameter changes via `config.yaml`, a fixed pipeline trains a LoRA adapter using mflux and evaluates it with CLIP scoring, and the LLM keeps or discards based on quantitative metrics -- looping indefinitely.

## How It Works

```
                    +-----------------+
                    |   LLM Agent     |
                    |  (Claude Code)  |
                    +--------+--------+
                             |
                     edits config.yaml
                             |
                             v
                    +------------------+
                    |   config.yaml    |
                    | (hyperparameters)|
                    +--------+---------+
                             |
                             v
+----------------------------------------------------------------+
|                     train.py (fixed)                            |
|                                                                |
|  1. Translate config.yaml --> mflux JSON                       |
|  2. mflux-train: train LoRA adapter (subprocess)               |
|  3. mflux-generate: produce 24 eval images (subprocess)        |
|  4. CLIP score eval images against reference embeddings         |
|  5. Print metrics to stdout                                    |
+----------------------------------------------------------------+
                             |
                             v
                    +------------------+
                    |   LLM reads      |
                    |   run.log        |
                    |   keep/discard   |
                    |   loop again     |
                    +------------------+
```

The LLM reads scores from `run.log`, decides whether the experiment improved (`clip_sim_centroid` delta >= 0.005), commits the result to git, and proposes the next experiment. Each cycle takes ~5-8 minutes. Overnight (~8 hours) yields ~60 experiments.

## Quick Start

### Prerequisites

- Apple Silicon Mac with 36GB+ unified memory
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd autoresearch-lora

# One-time setup: download model, compute CLIP embeddings, run smoke test
uv run prepare.py --images ./my-photos/
```

Place 5-20 reference images (512px+ resolution, .jpg/.png/.webp) in a directory and point `--images` at it.

### Run an Experiment

```bash
timeout 720 uv run train.py > run.log 2>&1
```

### Run the Autonomous Loop

The LLM agent uses `program.md` as its instruction set. Start it on a branch:

```bash
git checkout -b autoresearch-lora/my-experiment
# Then launch your LLM agent with program.md as context
```

The agent will edit `config.yaml`, run `train.py`, read scores, keep or discard, and repeat.

## Config Parameters

The LLM edits `config.yaml` to explore the search space. All other code is fixed.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rank` | int | 8 | LoRA rank (4-128). Controls adapter capacity. |
| `lr` | float | 3e-4 | AdamW optimizer learning rate. |
| `batch_size` | int | 1 | Training batch size. |
| `steps` | int | 9 | Denoising steps for the diffusion model. |
| `num_epochs` | int | 1 | Passes over the training dataset. |
| `quantize` | int/null | 4 | Base model quantization (3/4/5/6/8 or null). |
| `guidance` | float | 4.0 | Classifier-free guidance scale. |
| `trigger_word` | string | ohwx | Token that activates the LoRA. |
| `caption_template` | string | a photo of {trigger} | Template for training captions. |

## Scoring

Evaluation generates 24 images (6 prompts x 4 seeds) and scores them with CLIP:

- **clip_sim_centroid** (primary metric) -- mean cosine similarity of eval images vs. reference centroid
- **clip_sim_nn** -- mean cosine similarity vs. nearest reference image
- **neg_control** -- similarity score for images generated without the trigger (overfitting check)
- **score_stddev** -- standard deviation across eval images (consistency check)

Prompts 1-5 use the trigger word in various contexts. Prompt 6 is a negative control (landscape, no trigger).

## Project Structure

```
autoresearch-lora/
├── train.py              # Pipeline orchestrator (train, generate, score)
├── prepare.py            # One-time setup (model download, CLIP embeddings, smoke test)
├── config.yaml           # Hyperparameters (edited by LLM)
├── config_translator.py  # YAML config --> mflux JSON translation
├── score.py              # CLIP scoring module (mlx_clip)
├── eval_prompts.txt      # 6 eval prompts ({trigger} placeholder)
├── program.md            # LLM agent instructions
├── pyproject.toml        # Dependencies and project metadata
├── docs/
│   └── design.md         # Full architectural spec
└── tests/
    ├── test_config.py    # Config translator tests
    └── test_score.py     # Scoring module tests
```

Runtime artifacts (gitignored): `adapter/`, `training/`, `eval_images/`, `run.log`, `.mflux-train-config.json`

Cached data: `~/.cache/autoresearch-lora/` (images, CLIP embeddings, model)

## Tech Stack

- **Base model:** FLUX.2 Klein 4B via [mflux](https://github.com/filipstrand/mflux) 0.16.9
- **Scoring:** CLIP image-image similarity via [mlx_clip](https://github.com/harperreed/mlx_clip) (MLX-native, no PyTorch)
- **ML framework:** [MLX](https://github.com/ml-explore/mlx) (Apple Silicon optimized)
- **Runtime:** Python 3.11+, managed with [uv](https://docs.astral.sh/uv/)
- **Dependencies:** mflux, mlx, mlx_clip, numpy, pillow, pyyaml

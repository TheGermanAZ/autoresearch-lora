# autoresearch-lora Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous LoRA training loop for image generation on Apple Silicon that an LLM can run indefinitely.

**Architecture:** LLM edits config.yaml → fixed train.py orchestrates mflux subprocesses (train + generate) → CLIP image-image similarity scores results → LLM keeps/discards → loops forever. All pure MLX on Apple Silicon.

**Tech Stack:** Python 3.11+, mflux (LoRA training + inference), mlx + mlx_clip (CLIP scoring), pyyaml, numpy, pillow

**Spec:** `docs/design.md`

---

## File Structure

```
autoresearch-lora/
├── pyproject.toml          # Dependencies and project config
├── .gitignore              # Ignore eval_images/, adapter/, __pycache__, .cache
├── score.py                # CLIP scoring module (embeddings, cosine sim, centroid, NN)
├── config_translator.py    # Read config.yaml → mflux JSON config
├── train.py                # Subprocess orchestrator (fixed, not edited by LLM)
├── prepare.py              # One-time setup (download model, compute embeddings, smoke test)
├── program.md              # Instructions for the autonomous LLM
├── config.yaml             # Default config (written by prepare.py, edited by LLM)
├── eval_prompts.txt        # 6 eval prompts (written by prepare.py)
├── results.tsv             # Experiment log (initialized by LLM during setup)
├── reasoning.md            # LLM's research journal
├── reference_images/       # User's training images
├── eval_images/            # Generated eval images (24 per experiment)
├── adapter/                # mflux LoRA output (cleaned each run)
├── tests/
│   ├── test_score.py       # Unit tests for CLIP scoring logic
│   └── test_config.py      # Unit tests for config translation
└── docs/
    ├── design.md           # Approved spec
    ├── plans/              # This plan
    └── validation.md       # Pre-implementation validation findings
```

**Key decomposition:** `score.py` and `config_translator.py` are extracted from train.py so they can be unit-tested independently. train.py imports them and orchestrates subprocesses.

---

## Chunk 1: Pre-Implementation Validation

### Task 0: Validate mflux CLI end-to-end

This is a research task, not a code task. The findings inform all subsequent tasks.

**Files:**
- Create: `docs/validation.md`

- [ ] **Step 1: Install mflux and record version**

```bash
uv pip install "mflux>=0.16.0,<0.17"
mflux-train --help 2>&1 | head -20
python -c "import mflux; print(mflux.__version__)"
```

Record the exact version in `docs/validation.md`.

- [ ] **Step 2: Inspect mflux training config schema**

```bash
# Find the example train config in the installed package
python -c "import mflux; import os; print(os.path.dirname(mflux.__file__))"
```

Navigate to the dreambooth example config. Read the JSON schema. Record every field and its type in `docs/validation.md`.

- [ ] **Step 3: Train a dummy LoRA**

Create a minimal training config JSON with 5 steps. Use a single test image. Run:

```bash
mflux-train --train-config /tmp/test-train.json
```

Record: exact command, config JSON that worked, output directory, adapter file path and format.

- [ ] **Step 4: Generate an image with the trained LoRA**

```bash
mflux-generate-flux2 --base-model flux2-klein-base-4b \
  --lora-paths ./adapter/ \
  --prompt "a photo of ohwx" \
  --seed 42 --steps 4 --width 512 --height 512 \
  --output /tmp/test-gen.png
```

Record: exact command that worked, correct CLI entry point, correct --base-model value, --lora-paths format.

- [ ] **Step 5: Validate mlx_clip**

```python
import mlx_clip
# Load model, embed an image, compute cosine similarity
# Record: import path, API for embedding images, embedding shape
```

Record: exact API calls, embedding dimensions, any gotchas.

- [ ] **Step 6: Write docs/validation.md with all findings**

Document: mflux version pinned, JSON config schema, exact CLI commands, adapter output path, mlx_clip API, any surprises.

- [ ] **Step 7: Commit**

```bash
git add docs/validation.md
git commit -m "docs: record pre-implementation validation findings"
```

---

## Chunk 2: Project Scaffolding

### Task 1: Create pyproject.toml and .gitignore

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "autoresearch-lora"
version = "0.1.0"
description = "Autonomous LoRA training loop for image generation on Apple Silicon"
requires-python = ">=3.11"
dependencies = [
    "mflux>=0.16.0,<0.17",
    "mlx>=0.30.0,<0.32",
    "mlx-clip",
    "numpy",
    "pillow",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
```

Pin mflux to the exact version from validation findings.

- [ ] **Step 2: Write .gitignore**

```
__pycache__/
*.pyc
eval_images/
adapter/
*.npy
run.log
.cache/
```

- [ ] **Step 3: Install dependencies**

```bash
uv sync
uv sync --extra dev
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore uv.lock
git commit -m "chore: project scaffolding with dependencies"
```

---

## Chunk 3: CLIP Scoring Module

### Task 2: Build score.py with unit tests (TDD)

**Files:**
- Create: `tests/test_score.py`
- Create: `score.py`

- [ ] **Step 1: Write failing tests for cosine similarity**

```python
# tests/test_score.py
import numpy as np

def test_cosine_similarity_identical():
    """Identical vectors should have similarity 1.0"""
    from score import cosine_similarity
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6

def test_cosine_similarity_orthogonal():
    """Orthogonal vectors should have similarity 0.0"""
    from score import cosine_similarity
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6

def test_cosine_similarity_opposite():
    """Opposite vectors should have similarity -1.0"""
    from score import cosine_similarity
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_score.py -v
```

Expected: FAIL — `score` module not found.

- [ ] **Step 3: Implement cosine_similarity**

```python
# score.py
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_score.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Write failing tests for centroid and nearest-neighbor scoring**

```python
# tests/test_score.py (append)

def test_compute_centroid():
    from score import compute_centroid
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    centroid = compute_centroid(embeddings)
    expected = np.array([0.5, 0.5])
    np.testing.assert_array_almost_equal(centroid, expected)

def test_score_against_centroid():
    from score import score_against_centroid
    eval_embedding = np.array([0.5, 0.5])
    centroid = np.array([0.5, 0.5])
    sim = score_against_centroid(eval_embedding, centroid)
    assert abs(sim - 1.0) < 1e-6

def test_score_nearest_neighbor():
    from score import score_nearest_neighbor
    eval_embedding = np.array([1.0, 0.0])
    ref_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = score_nearest_neighbor(eval_embedding, ref_embeddings)
    assert abs(sim - 1.0) < 1e-6  # Nearest is identical

def test_aggregate_scores():
    from score import aggregate_scores
    # 5 prompts × 4 seeds = 20 centroid sims, 4 neg sims
    centroid_sims = [0.8] * 20
    nn_sims = [0.75] * 20
    neg_sims = [0.3] * 4
    result = aggregate_scores(centroid_sims, nn_sims, neg_sims, num_prompts=5, seeds_per_prompt=4)
    assert abs(result["clip_sim_centroid"] - 0.8) < 1e-6
    assert abs(result["clip_sim_nn"] - 0.75) < 1e-6
    assert abs(result["neg_control"] - 0.3) < 1e-6
    assert len(result["prompt_scores"]) == 5
    assert "score_stddev" in result
```

- [ ] **Step 6: Run tests to verify they fail**

```bash
uv run pytest tests/test_score.py -v
```

Expected: 4 new FAILs.

- [ ] **Step 7: Implement centroid, NN, and aggregation functions**

```python
# score.py (append)

def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Mean of embedding vectors."""
    return np.mean(embeddings, axis=0)

def score_against_centroid(eval_embedding: np.ndarray, centroid: np.ndarray) -> float:
    """Cosine similarity of one eval image against the reference centroid."""
    return cosine_similarity(eval_embedding, centroid)

def score_nearest_neighbor(eval_embedding: np.ndarray, ref_embeddings: np.ndarray) -> float:
    """Max cosine similarity of eval image against any reference image."""
    sims = [cosine_similarity(eval_embedding, ref) for ref in ref_embeddings]
    return max(sims)

def aggregate_scores(
    centroid_sims: list[float],
    nn_sims: list[float],
    neg_sims: list[float],
    num_prompts: int,
    seeds_per_prompt: int,
) -> dict:
    """Aggregate per-image scores into experiment-level metrics."""
    prompt_scores = []
    for i in range(num_prompts):
        start = i * seeds_per_prompt
        end = start + seeds_per_prompt
        prompt_scores.append(float(np.mean(centroid_sims[start:end])))

    return {
        "clip_sim_centroid": float(np.mean(centroid_sims)),
        "clip_sim_nn": float(np.mean(nn_sims)),
        "prompt_scores": prompt_scores,
        "score_stddev": float(np.std(centroid_sims)),
        "neg_control": float(np.mean(neg_sims)),
    }
```

- [ ] **Step 8: Run all tests**

```bash
uv run pytest tests/test_score.py -v
```

Expected: 7 PASS.

- [ ] **Step 9: Add CLIP embedding function**

This wraps mlx_clip. The exact API depends on validation findings (Task 0). Skeleton:

```python
# score.py (append)
from pathlib import Path
from PIL import Image

_clip_model = None

def load_clip():
    """Load mlx_clip model (cached)."""
    global _clip_model
    if _clip_model is None:
        import mlx_clip
        _clip_model = mlx_clip.load()  # Exact API from validation
    return _clip_model

def embed_image(image_path: Path) -> np.ndarray:
    """Compute CLIP embedding for a single image."""
    model = load_clip()
    img = Image.open(image_path).convert("RGB")
    # Exact API from validation findings
    embedding = model.encode_image(img)  # Placeholder — update from validation
    return np.array(embedding).flatten()
```

Update the exact mlx_clip API calls based on `docs/validation.md` findings.

- [ ] **Step 10: Commit**

```bash
git add score.py tests/test_score.py
git commit -m "feat: add CLIP scoring module with centroid + NN similarity"
```

---

## Chunk 4: Config Translation

### Task 3: Build config_translator.py with unit tests (TDD)

**Files:**
- Create: `tests/test_config.py`
- Create: `config_translator.py`

- [ ] **Step 1: Write failing tests for config loading**

```python
# tests/test_config.py
import tempfile
from pathlib import Path

def test_load_config():
    from config_translator import load_config
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("rank: 16\nalpha: 32\nlr: 0.0003\n")
        f.flush()
        config = load_config(Path(f.name))
    assert config["rank"] == 16
    assert config["alpha"] == 32
    assert config["lr"] == 0.0003

def test_load_config_invalid_yaml():
    from config_translator import load_config, ConfigError
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(": invalid: yaml: [")
        f.flush()
        try:
            load_config(Path(f.name))
            assert False, "Should have raised ConfigError"
        except ConfigError:
            pass
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_config.py -v
```

- [ ] **Step 3: Implement config loading**

```python
# config_translator.py
import yaml
from pathlib import Path

class ConfigError(Exception):
    pass

def load_config(path: Path) -> dict:
    """Load and validate config.yaml."""
    try:
        with open(path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}")
    if not isinstance(config, dict):
        raise ConfigError(f"Config must be a YAML mapping, got {type(config)}")
    return config
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_config.py -v
```

- [ ] **Step 5: Write failing test for mflux JSON translation**

```python
# tests/test_config.py (append)

def test_to_mflux_json():
    from config_translator import load_config, to_mflux_json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""
rank: 16
alpha: 32
lr: 0.0003
batch_size: 1
steps: 500
num_epochs: 1
quantize: 4
guidance: 4.0
target_layers: "default"
trigger_word: "ohwx"
caption_template: "a photo of {trigger}"
""")
        f.flush()
        config = load_config(Path(f.name))

    images_dir = Path("/tmp/test-images")
    adapter_dir = Path("/tmp/test-adapter")
    mflux_config = to_mflux_json(config, images_dir, adapter_dir)

    # Verify structure matches mflux's expected schema
    # Exact assertions depend on validation findings (Task 0)
    assert isinstance(mflux_config, dict)
    assert "steps" in mflux_config or "training_loop" in mflux_config
```

- [ ] **Step 6: Implement to_mflux_json**

The exact field mapping depends on validation findings. Skeleton:

```python
# config_translator.py (append)
import json

def to_mflux_json(config: dict, images_dir: Path, adapter_dir: Path) -> dict:
    """Translate config.yaml fields into mflux training JSON config."""
    # Structure from docs/validation.md
    # This is a placeholder — update with exact schema from validation
    mflux_config = {
        "model": "flux2-klein-base-4b",
        "seed": 42,
        "steps": config["steps"],
        "guidance": config["guidance"],
        "quantize": config["quantize"],
        "lora_rank": config["rank"],
        "output_dir": str(adapter_dir),
        "training_loop": {
            "batch_size": config["batch_size"],
            "num_epochs": config["num_epochs"],
            "learning_rate": config["lr"],
        },
        "examples": [],  # Populated from images_dir
    }

    # Build examples array from images
    for img_path in sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png")):
        caption = config["caption_template"].replace("{trigger}", config["trigger_word"])
        mflux_config["examples"].append({
            "image": str(img_path),
            "prompt": caption,
        })

    return mflux_config

def write_mflux_json(mflux_config: dict, output_path: Path):
    """Write mflux JSON config to disk."""
    with open(output_path, "w") as f:
        json.dump(mflux_config, f, indent=2)
```

- [ ] **Step 7: Run all config tests**

```bash
uv run pytest tests/test_config.py -v
```

- [ ] **Step 8: Commit**

```bash
git add config_translator.py tests/test_config.py
git commit -m "feat: add config.yaml to mflux JSON translator"
```

---

## Chunk 5: prepare.py

### Task 4: Build prepare.py

**Files:**
- Create: `prepare.py`

- [ ] **Step 1: Write prepare.py — argument parsing and directory setup**

```python
# prepare.py
"""
One-time setup for autoresearch-lora experiments.
Downloads model, validates images, computes reference embeddings, runs smoke test.

Usage: uv run prepare.py --images ./my-photos/
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CACHE_DIR = Path.home() / ".cache" / "autoresearch-lora"
IMAGES_DIR = CACHE_DIR / "images"
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}
MIN_IMAGES = 5
MAX_IMAGES = 20
MIN_RESOLUTION = 512
```

- [ ] **Step 2: Implement model download**

```python
def download_model():
    """Pre-download FLUX.2 Klein 4B via mflux-save."""
    print("Downloading FLUX.2 Klein 4B...")
    # Exact command from docs/validation.md
    result = subprocess.run(
        ["mflux-save", "--model", "flux2-klein-base-4b"],  # Update from validation
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"Error downloading model: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    print("Model downloaded.")
```

- [ ] **Step 3: Implement image validation and copy**

```python
def validate_and_copy_images(source_dir: Path):
    """Validate images and copy to cache directory."""
    from PIL import Image

    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    images = [p for p in source_dir.iterdir() if p.suffix.lower() in SUPPORTED_FORMATS]
    if len(images) < MIN_IMAGES:
        print(f"Warning: only {len(images)} images found (recommend {MIN_IMAGES}-{MAX_IMAGES})")
    if len(images) > MAX_IMAGES:
        print(f"Warning: {len(images)} images found (recommend max {MAX_IMAGES})")
    if len(images) == 0:
        print("Error: no supported images found", file=sys.stderr)
        sys.exit(1)

    # Validate resolution
    for img_path in images:
        with Image.open(img_path) as img:
            w, h = img.size
            if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
                print(f"Warning: {img_path.name} is {w}x{h} (min {MIN_RESOLUTION}x{MIN_RESOLUTION})")

    # Copy to cache
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        shutil.copy2(img_path, IMAGES_DIR / img_path.name)

    print(f"Copied {len(images)} images to {IMAGES_DIR}")
    return list(IMAGES_DIR.glob("*"))
```

- [ ] **Step 4: Implement reference embedding computation**

```python
def compute_reference_embeddings(image_paths: list[Path]):
    """Compute CLIP embeddings for all reference images."""
    from score import embed_image, compute_centroid

    print("Computing reference embeddings...")
    embeddings = []
    for path in sorted(image_paths):
        if path.suffix.lower() in SUPPORTED_FORMATS:
            emb = embed_image(path)
            embeddings.append(emb)

    embeddings_array = np.array(embeddings)
    centroid = compute_centroid(embeddings_array)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / "ref_centroid.npy", centroid)
    np.save(CACHE_DIR / "ref_embeddings.npy", embeddings_array)
    print(f"Saved centroid + {len(embeddings)} embeddings to {CACHE_DIR}")
```

- [ ] **Step 5: Implement default config and eval prompts**

```python
def write_default_config(project_dir: Path):
    """Write default config.yaml."""
    import yaml
    config = {
        "rank": 8,
        "alpha": 16,
        "lr": 3e-4,
        "batch_size": 1,
        "steps": 1000,
        "num_epochs": 1,
        "quantize": 4,
        "guidance": 4.0,
        "target_layers": "default",
        "trigger_word": "ohwx",
        "caption_template": "a photo of {trigger}",
    }
    config_path = project_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Default config written to {config_path}")

def write_eval_prompts(project_dir: Path):
    """Write default eval_prompts.txt."""
    prompts = [
        "a photo of {trigger}",
        "a photo of {trigger} in a coffee shop",
        "a watercolor painting of {trigger}",
        "{trigger} laughing, candid photo",
        "a close-up portrait of {trigger}, soft lighting",
        "a photo of a landscape at sunset",
    ]
    prompts_path = project_dir / "eval_prompts.txt"
    with open(prompts_path, "w") as f:
        f.write("\n".join(prompts) + "\n")
    print(f"Eval prompts written to {prompts_path}")
```

- [ ] **Step 6: Implement smoke test**

```python
def run_smoke_test(project_dir: Path):
    """Validate the full pipeline end-to-end."""
    print("\nRunning smoke test...")

    # (a) Train 5 steps
    print("  LoRA train (5 steps)...", end=" ", flush=True)
    # Build minimal config, run mflux-train with steps=5
    # Exact commands from docs/validation.md
    t0 = time.time()
    # ... subprocess call ...
    print(f"OK ({time.time()-t0:.1f}s)")

    # (b) Generate 1 image
    print("  Generate 1024x1024...", end=" ", flush=True)
    t0 = time.time()
    # ... subprocess call ...
    print(f"OK ({time.time()-t0:.1f}s)")

    # (c) CLIP score
    print("  CLIP scoring...", end=" ", flush=True)
    centroid = np.load(CACHE_DIR / "ref_centroid.npy")
    # ... embed and score ...
    print(f"OK (sim={sim:.3f})")

    # (d) Negative baseline
    print("  Negative baseline...", end=" ", flush=True)
    # ... generate landscape, score ...
    neg_score_path = CACHE_DIR / "neg_baseline.txt"
    with open(neg_score_path, "w") as f:
        f.write(f"{neg_sim:.6f}\n")
    print(f"{neg_sim:.3f} (saved)")

    # (e) Inference step calibration
    print("  Inference timing:")
    for steps in [8, 12, 20]:
        t0 = time.time()
        # ... generate 1 image at N steps ...
        elapsed = time.time() - t0
        total_est = elapsed * 24
        print(f"    {steps:2d} steps: {elapsed:.1f}s/image → 24 images = {total_est:.0f}s")
```

- [ ] **Step 7: Implement main and CLI**

```python
def main():
    parser = argparse.ArgumentParser(description="Setup autoresearch-lora")
    parser.add_argument("--images", required=True, help="Path to reference images directory")
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    source_dir = Path(args.images)

    download_model()
    image_paths = validate_and_copy_images(source_dir)
    compute_reference_embeddings(image_paths)
    write_default_config(project_dir)
    write_eval_prompts(project_dir)
    run_smoke_test(project_dir)

    print("\nReady! Run the autoresearch loop to begin experimenting.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Commit**

```bash
git add prepare.py
git commit -m "feat: add prepare.py for one-time setup"
```

---

## Chunk 6: train.py (Subprocess Orchestrator)

### Task 5: Build train.py

**Files:**
- Create: `train.py`

- [ ] **Step 1: Write train.py constants and imports**

```python
# train.py
"""
Autoresearch LoRA training pipeline. Single-run, single-file.
Reads config.yaml, trains a LoRA via mflux, generates eval images, scores them.

Usage: uv run train.py [--dry-run]
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Constants (fixed — do not modify)
TIME_BUDGET = 300          # 5 min training
TOTAL_TIMEOUT = 600        # 10 min total
HARD_CAP = 720             # 12 min subprocess kill
EVAL_RESOLUTION = 1024
EVAL_SEEDS = [42, 137, 256, 999]
EVAL_STEPS = 20            # Inference steps per image (may adjust after smoke test)
NUM_TRIGGER_PROMPTS = 5    # Prompts 1-5 (with trigger)
NUM_NEG_PROMPTS = 1        # Prompt 6 (negative control)
NEG_WARN_THRESHOLD = 0.45

CACHE_DIR = Path.home() / ".cache" / "autoresearch-lora"
PROJECT_DIR = Path(__file__).parent
ADAPTER_DIR = PROJECT_DIR / "adapter"
EVAL_DIR = PROJECT_DIR / "eval_images"
```

- [ ] **Step 2: Implement config loading and dry-run**

```python
def main():
    dry_run = "--dry-run" in sys.argv

    # Load config
    from config_translator import load_config, to_mflux_json, write_mflux_json, ConfigError
    try:
        config = load_config(PROJECT_DIR / "config.yaml")
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        print(f"Contents of config.yaml:")
        print((PROJECT_DIR / "config.yaml").read_text())
        sys.exit(1)

    # Translate to mflux JSON
    mflux_config = to_mflux_json(config, CACHE_DIR / "images", ADAPTER_DIR)
    mflux_json_path = PROJECT_DIR / ".mflux-train-config.json"
    write_mflux_json(mflux_config, mflux_json_path)

    if dry_run:
        print("DRY RUN — mflux training config:")
        print(json.dumps(mflux_config, indent=2))
        sys.exit(0)
```

- [ ] **Step 3: Implement training subprocess**

```python
    # Clean adapter directory
    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    # Clean eval directory
    if EVAL_DIR.exists():
        shutil.rmtree(EVAL_DIR)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Train
    print("Training LoRA...", flush=True)
    t_train_start = time.time()
    try:
        result = subprocess.run(
            ["mflux-train", "--train-config", str(mflux_json_path)],
            capture_output=True, text=True,
            timeout=TIME_BUDGET + 120,  # Training + model load overhead
        )
        if result.returncode != 0:
            print(f"TRAINING FAILED:\n{result.stderr[-500:]}", file=sys.stderr)
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("TRAINING TIMEOUT", file=sys.stderr)
        sys.exit(1)
    training_seconds = time.time() - t_train_start
```

- [ ] **Step 4: Implement eval image generation**

```python
    # Load eval prompts
    prompts = (PROJECT_DIR / "eval_prompts.txt").read_text().strip().split("\n")
    trigger = config.get("trigger_word", "ohwx")
    prompts = [p.replace("{trigger}", trigger) for p in prompts]

    # Generate 24 images (6 prompts × 4 seeds)
    print("Generating eval images...", flush=True)
    t_eval_start = time.time()
    image_paths = []
    for pi, prompt in enumerate(prompts):
        for seed in EVAL_SEEDS:
            out_path = EVAL_DIR / f"p{pi}_s{seed}.png"
            try:
                result = subprocess.run(
                    [
                        "mflux-generate-flux2",  # Update from validation
                        "--base-model", "flux2-klein-base-4b",
                        "--lora-paths", str(ADAPTER_DIR),
                        "--prompt", prompt,
                        "--seed", str(seed),
                        "--steps", str(EVAL_STEPS),
                        "--width", str(EVAL_RESOLUTION),
                        "--height", str(EVAL_RESOLUTION),
                        "--output", str(out_path),
                    ],
                    capture_output=True, text=True,
                    timeout=120,  # Per-image timeout
                )
                if result.returncode != 0:
                    print(f"  WARN: generation failed for p{pi}_s{seed}: {result.stderr[-200:]}")
                    continue
            except subprocess.TimeoutExpired:
                print(f"  WARN: generation timeout for p{pi}_s{seed}")
                continue
            image_paths.append((pi, seed, out_path))
    eval_seconds = time.time() - t_eval_start
```

- [ ] **Step 5: Implement CLIP scoring**

```python
    # Score
    from score import embed_image, score_against_centroid, score_nearest_neighbor, aggregate_scores

    centroid = np.load(CACHE_DIR / "ref_centroid.npy")
    ref_embeddings = np.load(CACHE_DIR / "ref_embeddings.npy")

    trigger_centroid_sims = []
    trigger_nn_sims = []
    neg_sims = []

    for pi, seed, img_path in image_paths:
        emb = embed_image(img_path)
        c_sim = score_against_centroid(emb, centroid)
        nn_sim = score_nearest_neighbor(emb, ref_embeddings)

        if pi < NUM_TRIGGER_PROMPTS:
            trigger_centroid_sims.append(c_sim)
            trigger_nn_sims.append(nn_sim)
        else:
            neg_sims.append(c_sim)

    scores = aggregate_scores(
        trigger_centroid_sims, trigger_nn_sims, neg_sims,
        num_prompts=NUM_TRIGGER_PROMPTS,
        seeds_per_prompt=len(EVAL_SEEDS),
    )
```

- [ ] **Step 6: Implement summary output**

```python
    # Get peak memory (macOS)
    import resource
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    # On macOS ru_maxrss is in bytes, on Linux it's in KB

    # Print summary
    prompt_scores_str = ", ".join(f"{s:.2f}" for s in scores["prompt_scores"])
    print("---")
    print(f"clip_sim_centroid:  {scores['clip_sim_centroid']:.6f}")
    print(f"clip_sim_nn:        {scores['clip_sim_nn']:.6f}")
    print(f"prompt_scores:      {prompt_scores_str}")
    print(f"score_stddev:       {scores['score_stddev']:.6f}")
    print(f"neg_control:        {scores['neg_control']:.6f}")
    print(f"peak_vram_mb:       {peak_mb:.1f}")
    print(f"training_seconds:   {training_seconds:.1f}")
    print(f"steps_completed:    {config.get('steps', 0)}")
    print(f"eval_seconds:       {eval_seconds:.1f}")
    print("---")

    if scores["neg_control"] > NEG_WARN_THRESHOLD:
        print(f"WARNING: neg_control ({scores['neg_control']:.3f}) > {NEG_WARN_THRESHOLD} — possible overfitting")

if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Commit**

```bash
git add train.py
git commit -m "feat: add train.py subprocess orchestrator"
```

---

## Chunk 7: program.md and Final Integration

### Task 6: Write program.md

**Files:**
- Create: `program.md`

- [ ] **Step 1: Write program.md**

Adapt from the experiment loop in `docs/design.md`. This is the file the LLM reads to know how to run the autonomous loop. Include: setup instructions, loop steps, keep/discard logic, NEVER STOP, strategy guidance, crash recovery, output format, results.tsv schema.

See `docs/design.md` sections "Experiment Loop" and "Strategy Guidance" for the full content.

- [ ] **Step 2: Commit**

```bash
git add program.md
git commit -m "feat: add program.md for autonomous LLM loop"
```

### Task 7: End-to-end integration test

- [ ] **Step 1: Gather 5-10 test images**

Use any 5+ images of a consistent subject (e.g., download from Unsplash). Place in a test directory.

- [ ] **Step 2: Run prepare.py**

```bash
uv run prepare.py --images ./test-images/
```

Verify: model downloads, images copied, embeddings computed, smoke test passes.

- [ ] **Step 3: Run train.py**

```bash
timeout 720 uv run train.py > run.log 2>&1
```

Verify: training completes, 24 eval images generated in `eval_images/`, scores printed.

- [ ] **Step 4: Verify output format**

```bash
grep "^clip_sim_centroid:" run.log
```

Should print a valid score.

- [ ] **Step 5: Run train.py --dry-run**

```bash
uv run train.py --dry-run
```

Verify: prints mflux JSON config without running training.

- [ ] **Step 6: Commit everything**

```bash
git add -A
git commit -m "feat: complete autoresearch-lora v0.1.0"
```

- [ ] **Step 7: Push**

```bash
git push origin main
```

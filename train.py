"""
Autoresearch LoRA training pipeline. Single-run, single-file.
Reads config.yaml, trains a LoRA via mflux, generates eval images, scores them.

Usage: uv run train.py [--dry-run]
"""

import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

# Constants (fixed — do not modify)
TIME_BUDGET = 300  # 5 min training
EVAL_RESOLUTION = 1024
EVAL_SEEDS = [42, 137, 256, 999]
EVAL_STEPS = 20  # Inference steps per image
NUM_TRIGGER_PROMPTS = 5  # Prompts 1-5 (with trigger)
NUM_NEG_PROMPTS = 1  # Prompt 6 (negative control)
NEG_WARN_THRESHOLD = 0.45

CACHE_DIR = Path.home() / ".cache" / "autoresearch-lora"
PROJECT_DIR = Path(__file__).parent
TRAINING_DIR = PROJECT_DIR / "training"
ADAPTER_DIR = PROJECT_DIR / "adapter"
EVAL_DIR = PROJECT_DIR / "eval_images"
DATA_DIR = CACHE_DIR / "train_data"


def extract_adapter_from_checkpoint(checkpoint_dir: Path) -> Path:
    """Find latest checkpoint ZIP and extract the adapter safetensors."""
    zip_files = sorted(checkpoint_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No checkpoint ZIP files in {checkpoint_dir}")

    latest_zip = zip_files[-1]
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(latest_zip) as zf:
        adapter_files = [
            n for n in zf.namelist() if n.endswith("_adapter.safetensors")
        ]
        if not adapter_files:
            raise FileNotFoundError(
                f"No adapter.safetensors in {latest_zip.name}"
            )
        zf.extract(adapter_files[0], ADAPTER_DIR)

    return ADAPTER_DIR / adapter_files[0]


def find_checkpoint_dir() -> Path:
    """Find the checkpoints directory created by mflux-train.

    mflux appends a timestamp to the output_path if it already exists,
    so we need to find the most recently created training* directory.
    """
    parent = TRAINING_DIR.parent
    candidates = sorted(
        [d for d in parent.glob("training*") if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for d in candidates:
        cp = d / "checkpoints"
        if cp.exists() and any(cp.glob("*.zip")):
            return cp
    # Fallback: check exact path
    cp = TRAINING_DIR / "checkpoints"
    if cp.exists():
        return cp
    raise FileNotFoundError("No checkpoints directory found after training")


def main():
    dry_run = "--dry-run" in sys.argv

    # Load config
    from config_translator import (
        ConfigError,
        load_config,
        prepare_data_dir,
        to_mflux_json,
        write_mflux_json,
    )

    try:
        config = load_config(PROJECT_DIR / "config.yaml")
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        print("Contents of config.yaml:")
        print((PROJECT_DIR / "config.yaml").read_text())
        sys.exit(1)

    # Prepare data directory (images + .txt captions)
    prepare_data_dir(config, CACHE_DIR / "images", DATA_DIR)

    # Translate to mflux JSON
    mflux_config = to_mflux_json(config, DATA_DIR, TRAINING_DIR)
    mflux_json_path = PROJECT_DIR / ".mflux-train-config.json"
    write_mflux_json(mflux_config, mflux_json_path)

    if dry_run:
        print("DRY RUN — mflux training config:")
        print(json.dumps(mflux_config, indent=2))
        sys.exit(0)

    # Clean previous training artifacts
    for d in PROJECT_DIR.parent.glob("training*"):
        if d.is_dir() and d.name.startswith("training"):
            shutil.rmtree(d, ignore_errors=True)
    if TRAINING_DIR.exists():
        shutil.rmtree(TRAINING_DIR, ignore_errors=True)
    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    if EVAL_DIR.exists():
        shutil.rmtree(EVAL_DIR)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # --- TRAIN ---
    print("Training LoRA...", flush=True)
    t_train_start = time.time()
    try:
        result = subprocess.run(
            ["mflux-train", "--config", str(mflux_json_path)],
            capture_output=True,
            text=True,
            timeout=TIME_BUDGET + 120,  # Training + model load overhead
        )
        if result.returncode != 0:
            print(f"TRAINING FAILED:\n{result.stderr[-500:]}", file=sys.stderr)
            sys.exit(1)
    except subprocess.TimeoutExpired:
        print("TRAINING TIMEOUT", file=sys.stderr)
        sys.exit(1)
    training_seconds = time.time() - t_train_start
    print(f"Training complete ({training_seconds:.1f}s)")

    # Extract adapter from checkpoint
    try:
        checkpoint_dir = find_checkpoint_dir()
        adapter_path = extract_adapter_from_checkpoint(checkpoint_dir)
    except FileNotFoundError as e:
        print(f"ADAPTER EXTRACTION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Adapter: {adapter_path}")

    # --- GENERATE EVAL IMAGES ---
    prompts = (PROJECT_DIR / "eval_prompts.txt").read_text().strip().split("\n")
    trigger = config.get("trigger_word", "ohwx")
    prompts = [p.replace("{trigger}", trigger) for p in prompts]

    print("Generating eval images...", flush=True)
    t_eval_start = time.time()
    image_paths = []
    quantize_args = []
    q = config.get("quantize")
    if q is not None:
        quantize_args = ["--quantize", str(q)]

    for pi, prompt in enumerate(prompts):
        for seed in EVAL_SEEDS:
            out_path = EVAL_DIR / f"p{pi}_s{seed}.png"
            cmd = [
                "mflux-generate",
                "--model", "flux2-klein-base-4b",
                "--prompt", prompt,
                "--seed", str(seed),
                "--steps", str(EVAL_STEPS),
                "--width", str(EVAL_RESOLUTION),
                "--height", str(EVAL_RESOLUTION),
                "--output", str(out_path),
            ] + quantize_args
            # Only add LoRA for trigger prompts (not negative control)
            if pi < NUM_TRIGGER_PROMPTS:
                cmd.extend(["--lora-paths", str(adapter_path)])
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    print(
                        f"  WARN: generation failed for p{pi}_s{seed}: {result.stderr[-200:]}"
                    )
                    continue
            except subprocess.TimeoutExpired:
                print(f"  WARN: generation timeout for p{pi}_s{seed}")
                continue
            if out_path.exists():
                image_paths.append((pi, seed, out_path))
    eval_seconds = time.time() - t_eval_start
    print(f"Generated {len(image_paths)} images ({eval_seconds:.1f}s)")

    # --- CLIP SCORING ---
    from score import (
        aggregate_scores,
        embed_image,
        score_against_centroid,
        score_nearest_neighbor,
    )

    centroid = np.load(CACHE_DIR / "ref_centroid.npy")
    ref_embeddings = np.load(CACHE_DIR / "ref_embeddings.npy")

    trigger_centroid_sims = []
    trigger_nn_sims = []
    neg_sims = []

    print("Scoring...", flush=True)
    for pi, seed, img_path in image_paths:
        emb = embed_image(img_path)
        c_sim = score_against_centroid(emb, centroid)
        nn_sim = score_nearest_neighbor(emb, ref_embeddings)

        if pi < NUM_TRIGGER_PROMPTS:
            trigger_centroid_sims.append(c_sim)
            trigger_nn_sims.append(nn_sim)
        else:
            neg_sims.append(c_sim)

    if not trigger_centroid_sims:
        print("ERROR: No trigger prompt images scored", file=sys.stderr)
        sys.exit(1)

    scores = aggregate_scores(
        trigger_centroid_sims,
        trigger_nn_sims,
        neg_sims if neg_sims else [0.0],
        num_prompts=NUM_TRIGGER_PROMPTS,
        seeds_per_prompt=len(EVAL_SEEDS),
    )

    # --- SUMMARY OUTPUT ---
    # Get peak memory (macOS: ru_maxrss in bytes)
    import resource

    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak_bytes / (1024 * 1024)

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
        print(
            f"WARNING: neg_control ({scores['neg_control']:.3f}) > {NEG_WARN_THRESHOLD} — possible overfitting"
        )


if __name__ == "__main__":
    main()

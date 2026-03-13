"""
Autoresearch LoRA training pipeline.
Reads config.yaml, trains a LoRA via mflux, generates eval images, scores them.

Modes:
  uv run train.py              Single experiment from config.yaml
  uv run train.py --batch      Run batch.yaml experiments sequentially, rank results
  uv run train.py --screen     Cheap 1-epoch sanity check (no eval images)
  uv run train.py --dry-run    Print mflux config without running
"""

import json
import resource
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml

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


def clean_artifacts():
    """Remove training artifacts from previous runs."""
    for d in PROJECT_DIR.glob("training*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    if TRAINING_DIR.exists():
        shutil.rmtree(TRAINING_DIR, ignore_errors=True)
    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    if EVAL_DIR.exists():
        shutil.rmtree(EVAL_DIR)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)


def train_lora(config: dict, mflux_json_path: Path) -> tuple[float, Path, int]:
    """Train LoRA and return (training_seconds, adapter_path, iterations_completed)."""
    from config_translator import prepare_data_dir, to_mflux_json, write_mflux_json

    prepare_data_dir(config, CACHE_DIR / "images", DATA_DIR)
    mflux_config = to_mflux_json(config, DATA_DIR, TRAINING_DIR)
    write_mflux_json(mflux_config, mflux_json_path)

    clean_artifacts()

    print("Training LoRA...", flush=True)
    t_train_start = time.time()
    try:
        result = subprocess.run(
            ["mflux-train", "--config", str(mflux_json_path)],
            capture_output=True,
            text=True,
            timeout=TIME_BUDGET + 300,
        )
        if result.returncode != 0:
            raise RuntimeError(f"TRAINING FAILED:\n{result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("TRAINING TIMEOUT")
    training_seconds = time.time() - t_train_start
    print(f"Training complete ({training_seconds:.1f}s)")

    checkpoint_dir = find_checkpoint_dir()
    adapter_path = extract_adapter_from_checkpoint(checkpoint_dir)
    zip_files = sorted(checkpoint_dir.glob("*.zip"))
    iterations_completed = int(zip_files[-1].stem.split("_")[0]) if zip_files else 0
    print(f"Adapter: {adapter_path}")

    return training_seconds, adapter_path, iterations_completed


def generate_and_score(config: dict, adapter_path: Path) -> tuple[dict, float]:
    """Generate eval images, score them, return (scores_dict, eval_seconds)."""
    from score import (
        aggregate_scores,
        embed_image,
        score_against_centroid,
        score_nearest_neighbor,
    )

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
            if pi < NUM_TRIGGER_PROMPTS:
                cmd.extend(["--lora-paths", str(adapter_path)])
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120,
                )
                if result.returncode != 0:
                    print(f"  WARN: generation failed for p{pi}_s{seed}: {result.stderr[-200:]}")
                    continue
            except subprocess.TimeoutExpired:
                print(f"  WARN: generation timeout for p{pi}_s{seed}")
                continue
            if out_path.exists():
                image_paths.append((pi, seed, out_path))
    eval_seconds = time.time() - t_eval_start
    print(f"Generated {len(image_paths)} images ({eval_seconds:.1f}s)")

    # Score
    centroid = np.load(CACHE_DIR / "ref_centroid.npy")
    ref_embeddings = np.load(CACHE_DIR / "ref_embeddings.npy")

    per_prompt_centroid = {}
    per_prompt_nn = {}
    neg_sims = []

    print("Scoring...", flush=True)
    for pi, seed, img_path in image_paths:
        emb = embed_image(img_path)
        c_sim = score_against_centroid(emb, centroid)
        nn_sim = score_nearest_neighbor(emb, ref_embeddings)

        if pi < NUM_TRIGGER_PROMPTS:
            per_prompt_centroid.setdefault(pi, []).append(c_sim)
            per_prompt_nn.setdefault(pi, []).append(nn_sim)
        else:
            neg_sims.append(c_sim)

    if not per_prompt_centroid:
        raise RuntimeError("No trigger prompt images scored")

    scores = aggregate_scores(
        per_prompt_centroid, per_prompt_nn, neg_sims,
        num_prompts=NUM_TRIGGER_PROMPTS,
    )
    return scores, eval_seconds


def run_experiment(config: dict, tag: str = "") -> dict:
    """Run a full experiment: train + generate + score. Returns result dict."""
    label = f"[{tag}] " if tag else ""
    print(f"\n{'='*60}")
    print(f"{label}Starting experiment")
    print(f"{'='*60}")

    mflux_json_path = PROJECT_DIR / ".mflux-train-config.json"

    try:
        training_seconds, adapter_path, iterations = train_lora(config, mflux_json_path)
        scores, eval_seconds = generate_and_score(config, adapter_path)
    except RuntimeError as e:
        print(f"{label}FAILED: {e}", file=sys.stderr)
        return {
            "tag": tag,
            "status": "crash",
            "error": str(e),
            "clip_sim_centroid": 0.0,
            "config": config,
        }

    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak_bytes / (1024 * 1024)

    result = {
        "tag": tag,
        "status": "ok",
        "clip_sim_centroid": scores["clip_sim_centroid"],
        "clip_sim_nn": scores["clip_sim_nn"],
        "prompt_scores": scores["prompt_scores"],
        "score_stddev": scores["score_stddev"],
        "neg_control": scores["neg_control"],
        "peak_vram_mb": peak_mb,
        "training_seconds": training_seconds,
        "iterations_completed": iterations,
        "eval_seconds": eval_seconds,
        "config": config,
    }

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
    print(f"steps_completed:    {iterations}")
    print(f"eval_seconds:       {eval_seconds:.1f}")
    print("---")

    if scores["neg_control"] > NEG_WARN_THRESHOLD:
        print(f"WARNING: neg_control ({scores['neg_control']:.3f}) > {NEG_WARN_THRESHOLD} — possible overfitting")

    return result


def screen_experiment(config: dict, tag: str = "") -> dict:
    """Cheap screening: 1-epoch training only, no eval images. Checks for crashes."""
    from config_translator import prepare_data_dir, to_mflux_json, write_mflux_json

    label = f"[{tag}] " if tag else ""
    print(f"{label}Screening...", end=" ", flush=True)

    screen_config = dict(config)
    screen_config["num_epochs"] = 1

    prepare_data_dir(screen_config, CACHE_DIR / "images", DATA_DIR)
    mflux_config = to_mflux_json(screen_config, DATA_DIR, TRAINING_DIR)
    mflux_json_path = PROJECT_DIR / ".mflux-train-config.json"
    write_mflux_json(mflux_config, mflux_json_path)

    clean_artifacts()

    t0 = time.time()
    try:
        result = subprocess.run(
            ["mflux-train", "--config", str(mflux_json_path)],
            capture_output=True, text=True, timeout=180,
        )
        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"CRASH ({elapsed:.0f}s)")
            return {"tag": tag, "status": "crash", "error": result.stderr[-300:], "seconds": elapsed}
        print(f"OK ({elapsed:.0f}s)")
        return {"tag": tag, "status": "ok", "seconds": elapsed}
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"TIMEOUT ({elapsed:.0f}s)")
        return {"tag": tag, "status": "timeout", "seconds": elapsed}


def load_batch_configs() -> list[tuple[str, dict]]:
    """Load batch.yaml, merge each experiment with config.yaml base."""
    from config_translator import ConfigError, load_config

    base_config = load_config(PROJECT_DIR / "config.yaml")
    batch_path = PROJECT_DIR / "batch.yaml"
    if not batch_path.exists():
        print("ERROR: batch.yaml not found", file=sys.stderr)
        sys.exit(1)

    with open(batch_path) as f:
        batch = yaml.safe_load(f)

    if not batch or "experiments" not in batch:
        print("ERROR: batch.yaml must have an 'experiments' list", file=sys.stderr)
        sys.exit(1)

    configs = []
    for exp in batch["experiments"]:
        tag = exp.pop("tag", f"exp{len(configs)}")
        merged = dict(base_config)
        merged.update(exp)
        configs.append((tag, merged))

    return configs


def run_batch(screen_mode: bool = False):
    """Run all experiments in batch.yaml sequentially, rank and report."""
    configs = load_batch_configs()
    print(f"\n{'='*60}")
    print(f"BATCH: {len(configs)} experiments" + (" (screen)" if screen_mode else ""))
    print(f"{'='*60}")

    results = []
    for tag, config in configs:
        if screen_mode:
            r = screen_experiment(config, tag)
        else:
            r = run_experiment(config, tag)
        results.append(r)

    # Report
    print(f"\n{'='*60}")
    print(f"BATCH RESULTS — {len(results)} experiments")
    print(f"{'='*60}")

    if screen_mode:
        for r in results:
            status = r["status"]
            secs = r.get("seconds", 0)
            print(f"  [{r['tag']}] {status} ({secs:.0f}s)")
        passed = [r for r in results if r["status"] == "ok"]
        print(f"\n{len(passed)}/{len(results)} passed screening")
        return

    # Sort by clip_sim_centroid (descending), crashes last
    ranked = sorted(
        results,
        key=lambda r: r.get("clip_sim_centroid", -1) if r["status"] == "ok" else -1,
        reverse=True,
    )

    for i, r in enumerate(ranked):
        tag = r["tag"]
        if r["status"] != "ok":
            print(f"  [{tag}] CRASH: {r.get('error', 'unknown')[:80]}")
            continue
        marker = " <-- BEST" if i == 0 else ""
        print(
            f"  [{tag}] clip_centroid={r['clip_sim_centroid']:.3f}"
            f"  clip_nn={r['clip_sim_nn']:.3f}"
            f"  neg={r['neg_control']:.3f}"
            f"  train={r['training_seconds']:.0f}s{marker}"
        )

    # Output best result in machine-readable format
    best = ranked[0] if ranked and ranked[0]["status"] == "ok" else None
    if best:
        print(f"\nBEST: [{best['tag']}] clip_sim_centroid={best['clip_sim_centroid']:.6f}")
        # Output config as JSON for the LLM to parse and apply
        best_cfg = {k: v for k, v in best["config"].items()}
        print(f"best_config: {json.dumps(best_cfg)}")
    else:
        print("\nNo successful experiments in batch.")


def main():
    dry_run = "--dry-run" in sys.argv
    batch_mode = "--batch" in sys.argv
    screen_mode = "--screen" in sys.argv

    if batch_mode:
        run_batch(screen_mode=screen_mode)
        return

    # Single experiment mode
    from config_translator import ConfigError, load_config, to_mflux_json, write_mflux_json

    try:
        config = load_config(PROJECT_DIR / "config.yaml")
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        print("Contents of config.yaml:")
        print((PROJECT_DIR / "config.yaml").read_text())
        sys.exit(1)

    if dry_run:
        from config_translator import prepare_data_dir
        prepare_data_dir(config, CACHE_DIR / "images", DATA_DIR)
        mflux_config = to_mflux_json(config, DATA_DIR, TRAINING_DIR)
        print("DRY RUN — mflux training config:")
        print(json.dumps(mflux_config, indent=2))
        sys.exit(0)

    if screen_mode:
        r = screen_experiment(config)
        sys.exit(0 if r["status"] == "ok" else 1)

    result = run_experiment(config)
    sys.exit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()

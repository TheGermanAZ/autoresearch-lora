"""
One-time setup for autoresearch-lora experiments.
Downloads model, validates images, computes reference embeddings, runs smoke test.

Usage: uv run prepare.py --images ./my-photos/
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml

CACHE_DIR = Path.home() / ".cache" / "autoresearch-lora"
IMAGES_DIR = CACHE_DIR / "images"
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp"}
MIN_IMAGES = 5
MAX_IMAGES = 20
MIN_RESOLUTION = 512


def download_model():
    """Pre-download FLUX.2 Klein base 4B via mflux (first generate triggers download)."""
    print("Checking FLUX.2 Klein base 4B model availability...")
    # mflux-save requires --path; we'll just verify the model name is valid
    # The actual download happens on first training run, but we can trigger
    # a quick validation with mflux-train --dry-run
    print("Model will be downloaded on first training run.")
    print("(mflux handles model caching via HuggingFace hub)")


def validate_and_copy_images(source_dir: Path) -> list[Path]:
    """Validate images and copy to cache directory."""
    from PIL import Image

    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        print(f"Error: {source_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    images = [
        p for p in source_dir.iterdir() if p.suffix.lower() in SUPPORTED_FORMATS
    ]
    if len(images) == 0:
        print("Error: no supported images found", file=sys.stderr)
        sys.exit(1)
    if len(images) < MIN_IMAGES:
        print(
            f"Warning: only {len(images)} images found (recommend {MIN_IMAGES}-{MAX_IMAGES})"
        )
    if len(images) > MAX_IMAGES:
        print(
            f"Warning: {len(images)} images found (recommend max {MAX_IMAGES})"
        )

    # Validate resolution
    for img_path in images:
        with Image.open(img_path) as img:
            w, h = img.size
            if w < MIN_RESOLUTION or h < MIN_RESOLUTION:
                print(
                    f"Warning: {img_path.name} is {w}x{h} (min {MIN_RESOLUTION}x{MIN_RESOLUTION})"
                )

    # Clear and re-create cache to avoid stale images from previous runs
    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for img_path in images:
        shutil.copy2(img_path, IMAGES_DIR / img_path.name)

    print(f"Copied {len(images)} images to {IMAGES_DIR}")
    return sorted(IMAGES_DIR.glob("*"))


def compute_reference_embeddings(image_paths: list[Path]):
    """Compute CLIP embeddings for all reference images."""
    from score import compute_centroid, embed_image

    print("Computing reference embeddings...")
    embeddings = []
    for path in sorted(image_paths):
        if path.suffix.lower() in SUPPORTED_FORMATS:
            print(f"  Embedding {path.name}...", end=" ", flush=True)
            emb = embed_image(path)
            embeddings.append(emb)
            print("OK")

    embeddings_array = np.array(embeddings)
    centroid = compute_centroid(embeddings_array)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / "ref_centroid.npy", centroid)
    np.save(CACHE_DIR / "ref_embeddings.npy", embeddings_array)
    print(f"Saved centroid + {len(embeddings)} embeddings to {CACHE_DIR}")


def write_default_config(project_dir: Path):
    """Write default config.yaml."""
    config = {
        "rank": 8,
        "lr": 3e-4,
        "batch_size": 1,
        "steps": 9,
        "num_epochs": 1,
        "quantize": 4,
        "guidance": 4.0,
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


def run_smoke_test(project_dir: Path):
    """Validate the full pipeline end-to-end with minimal training."""
    from config_translator import prepare_data_dir, to_mflux_json, write_mflux_json

    print("\n--- Smoke Test ---")

    config = yaml.safe_load((project_dir / "config.yaml").read_text())

    # Prepare data directory for training
    data_dir = CACHE_DIR / "train_data"
    prepare_data_dir(config, IMAGES_DIR, data_dir)

    # (a) Train 1 epoch with minimal steps
    print("  [1/5] LoRA train (1 epoch, steps=9)...", end=" ", flush=True)
    smoke_config = dict(config)
    smoke_config["num_epochs"] = 1
    smoke_config["steps"] = 9

    training_dir = project_dir / "training"
    mflux_config = to_mflux_json(smoke_config, data_dir, training_dir)
    mflux_json_path = project_dir / ".mflux-train-config.json"
    write_mflux_json(mflux_config, mflux_json_path)

    t0 = time.time()
    result = subprocess.run(
        ["mflux-train", "--config", str(mflux_json_path)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"FAIL\n  {result.stderr[-500:]}")
        sys.exit(1)
    print(f"OK ({time.time() - t0:.1f}s)")

    # Find the latest checkpoint (same logic as train.py: sort by mtime, require .zip)
    checkpoint_dir = None
    parent = training_dir.parent
    candidates = sorted(
        [d for d in parent.glob("training*") if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for d in candidates:
        cp = d / "checkpoints"
        if cp.exists() and any(cp.glob("*.zip")):
            checkpoint_dir = cp
            training_dir = d
            break
    if not checkpoint_dir:
        checkpoint_dir = training_dir / "checkpoints"

    if not checkpoint_dir.exists() or not any(checkpoint_dir.glob("*.zip")):
        print("  FAIL: no checkpoints directory with ZIP files found")
        sys.exit(1)

    # Extract adapter from latest checkpoint ZIP
    zip_files = sorted(checkpoint_dir.glob("*.zip"))
    if not zip_files:
        print(f"  FAIL: no checkpoint ZIP files found in {checkpoint_dir}")
        sys.exit(1)
    latest_zip = zip_files[-1]
    adapter_dir = project_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Find the safetensors file inside the ZIP
    with zipfile.ZipFile(latest_zip) as zf:
        adapter_files = [n for n in zf.namelist() if n.endswith("_adapter.safetensors")]
        if not adapter_files:
            print(f"  FAIL: no adapter.safetensors in {latest_zip}")
            sys.exit(1)
        zf.extract(adapter_files[0], adapter_dir)
    adapter_path = adapter_dir / adapter_files[0]
    print(f"  Adapter extracted: {adapter_path}")

    # (b) Generate 1 image
    print("  [2/5] Generate 1024x1024...", end=" ", flush=True)
    t0 = time.time()
    gen_output = project_dir / "eval_images"
    gen_output.mkdir(parents=True, exist_ok=True)
    test_img = gen_output / "smoke_test.png"
    result = subprocess.run(
        [
            "mflux-generate",
            "--model", "flux2-klein-base-4b",
            "--lora-paths", str(adapter_path),
            "--prompt", "a photo of ohwx",
            "--seed", "42",
            "--steps", "4",
            "--width", "1024",
            "--height", "1024",
            "--output", str(test_img),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"FAIL\n  {result.stderr[-500:]}")
        sys.exit(1)
    if not test_img.exists():
        print("FAIL (no output image)")
        sys.exit(1)
    print(f"OK ({time.time() - t0:.1f}s)")

    # (c) CLIP score
    print("  [3/5] CLIP scoring...", end=" ", flush=True)
    from score import embed_image, score_against_centroid

    centroid = np.load(CACHE_DIR / "ref_centroid.npy")
    emb = embed_image(test_img)
    sim = score_against_centroid(emb, centroid)
    print(f"OK (sim={sim:.3f})")

    # (d) Negative baseline
    print("  [4/5] Negative baseline...", end=" ", flush=True)
    neg_img = gen_output / "smoke_neg.png"
    result = subprocess.run(
        [
            "mflux-generate",
            "--model", "flux2-klein-base-4b",
            "--prompt", "a photo of a landscape at sunset",
            "--seed", "42",
            "--steps", "4",
            "--width", "1024",
            "--height", "1024",
            "--output", str(neg_img),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        print(f"FAIL\n  {result.stderr[-500:]}")
    else:
        neg_emb = embed_image(neg_img)
        neg_sim = score_against_centroid(neg_emb, centroid)
        neg_score_path = CACHE_DIR / "neg_baseline.txt"
        neg_score_path.write_text(f"{neg_sim:.6f}\n")
        print(f"{neg_sim:.3f} (saved)")

    # (e) Inference step calibration
    print("  [5/5] Inference timing:")
    for steps in [4, 8, 20]:
        t0 = time.time()
        cal_img = gen_output / f"smoke_cal_{steps}.png"
        subprocess.run(
            [
                "mflux-generate",
                "--model", "flux2-klein-base-4b",
                "--lora-paths", str(adapter_path),
                "--prompt", "a photo of ohwx",
                "--seed", "42",
                "--steps", str(steps),
                "--width", "1024",
                "--height", "1024",
                "--output", str(cal_img),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.time() - t0
        total_est = elapsed * 24
        print(
            f"    {steps:2d} steps: {elapsed:.1f}s/image → 24 images = {total_est:.0f}s"
        )

    # Cleanup smoke test artifacts
    shutil.rmtree(training_dir, ignore_errors=True)
    shutil.rmtree(adapter_dir, ignore_errors=True)
    shutil.rmtree(gen_output, ignore_errors=True)

    print("\n--- Smoke Test Complete ---")


def main():
    parser = argparse.ArgumentParser(description="Setup autoresearch-lora")
    parser.add_argument(
        "--images", required=True, help="Path to reference images directory"
    )
    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help="Skip the full smoke test (useful if model not downloaded yet)",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    source_dir = Path(args.images)

    download_model()
    image_paths = validate_and_copy_images(source_dir)
    compute_reference_embeddings(image_paths)
    write_default_config(project_dir)
    write_eval_prompts(project_dir)

    if not args.skip_smoke_test:
        run_smoke_test(project_dir)

    print("\nReady! Run the autoresearch loop to begin experimenting.")


if __name__ == "__main__":
    main()

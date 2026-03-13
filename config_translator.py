"""Translate config.yaml into mflux training JSON config.

Reads the YAML config the LLM edits and produces the JSON that mflux-train expects.
Also handles preparing the data directory with images + .txt caption files.
"""

import json
import shutil
from pathlib import Path

import yaml


class ConfigError(Exception):
    pass


# Default LoRA targets for Klein 4B attention layers (dual-stream blocks 0-4)
KLEIN_4B_DEFAULT_TARGETS = [
    {
        "module_path": "transformer_blocks.{block}.attn.to_q",
        "blocks": {"start": 0, "end": 5},
    },
    {
        "module_path": "transformer_blocks.{block}.attn.to_k",
        "blocks": {"start": 0, "end": 5},
    },
    {
        "module_path": "transformer_blocks.{block}.attn.to_v",
        "blocks": {"start": 0, "end": 5},
    },
    {
        "module_path": "transformer_blocks.{block}.attn.to_out",
        "blocks": {"start": 0, "end": 5},
    },
]


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


def to_mflux_json(config: dict, data_dir: Path, output_dir: Path) -> dict:
    """Translate config.yaml fields into mflux training JSON config."""
    rank = config.get("rank", 8)

    # Build LoRA targets with the configured rank
    targets = []
    for t in KLEIN_4B_DEFAULT_TARGETS:
        target = dict(t)
        target["rank"] = rank
        targets.append(target)

    quantize_val = config.get("quantize")
    if quantize_val is not None:
        quantize_val = int(quantize_val)

    return {
        "model": "flux2-klein-base-4b",
        "data": str(data_dir),
        "seed": 42,
        "steps": config.get("steps", 9),
        "guidance": config.get("guidance", 4.0),
        "quantize": quantize_val,
        "max_resolution": 1024,
        "low_ram": False,
        "training_loop": {
            "num_epochs": config.get("num_epochs", 1),
            "batch_size": config.get("batch_size", 1),
            "timestep_low": 0,
            "timestep_high": None,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": config.get("lr", 3e-4),
        },
        "checkpoint": {
            "save_frequency": config.get("num_epochs", 1),
            "output_path": str(output_dir),
        },
        "monitoring": None,
        "lora_layers": {
            "targets": targets,
        },
    }


def write_mflux_json(mflux_config: dict, output_path: Path):
    """Write mflux JSON config to disk."""
    with open(output_path, "w") as f:
        json.dump(mflux_config, f, indent=2)


def prepare_data_dir(config: dict, source_images_dir: Path, data_dir: Path):
    """Prepare mflux data directory: copy images and create caption .txt files.

    mflux auto-discovers images in the data directory and expects a matching
    .txt file for each image containing the caption/prompt.
    """
    # Clear stale data from previous runs before populating
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    trigger = config.get("trigger_word", "ohwx")
    template = config.get("caption_template", "a photo of {trigger}")
    caption = template.replace("{trigger}", trigger)

    supported = {".jpg", ".jpeg", ".png", ".webp"}
    for img_path in sorted(source_images_dir.iterdir()):
        if img_path.suffix.lower() not in supported:
            continue
        # Copy image
        dest_img = data_dir / img_path.name
        shutil.copy2(img_path, dest_img)
        # Write matching caption
        txt_path = dest_img.with_suffix(".txt")
        txt_path.write_text(caption + "\n")

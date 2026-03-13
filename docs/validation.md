# Pre-Implementation Validation Findings

## mflux v0.16.9

### CLI Entry Points

- `mflux-train --config <path.json> [--model flux2-klein-base-4b] [--quantize 4] [--dry-run]`
- `mflux-generate --model flux2-klein-base-4b --prompt "text" --seed 42 --steps 20 --width 1024 --height 1024 --output out.png [--lora-paths adapter.safetensors] [--quantize 4]`
- `mflux-save --model flux2-klein-base-4b --path ./saved-model/ [--quantize 4]`

### Training Config JSON Schema

```json
{
  "model": "flux2-klein-base-4b",
  "data": "./data/",
  "seed": 42,
  "steps": 9,
  "guidance": 4.0,
  "quantize": 4,
  "max_resolution": 1024,
  "low_ram": false,
  "training_loop": {
    "num_epochs": 1,
    "batch_size": 1,
    "timestep_low": 0,
    "timestep_high": null
  },
  "optimizer": {
    "name": "AdamW",
    "learning_rate": 1e-4
  },
  "checkpoint": {
    "save_frequency": 100,
    "output_path": "./training"
  },
  "monitoring": null,
  "lora_layers": {
    "targets": [
      {
        "module_path": "transformer_blocks.{block}.attn.to_q",
        "blocks": { "start": 0, "end": 5 },
        "rank": 8
      }
    ]
  }
}
```

### Field Details

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Model name or HF repo. Use `"flux2-klein-base-4b"`. |
| `data` | string | **Directory path** (not a list). Auto-discovers images + matching `.txt` prompt files. |
| `seed` | int | Random seed for training noise. |
| `steps` | int | Total denoising steps for the diffusion process (not training iterations). |
| `guidance` | float | CFG guidance during training (0.0 = no guidance). |
| `quantize` | int\|null | Base model quantization: 3, 4, 5, 6, 8, or null. |
| `max_resolution` | int\|null | Cap on largest image side. |
| `low_ram` | bool | Enable disk-backed cache for reduced RAM. |
| `training_loop.num_epochs` | int | Number of passes over the dataset. |
| `training_loop.batch_size` | int | Batch size. |
| `training_loop.timestep_low` | int | Min timestep for training noise (default 0). |
| `training_loop.timestep_high` | int\|null | Max timestep (default = steps). |
| `optimizer.name` | string | `"Adam"` or `"AdamW"`. |
| `optimizer.learning_rate` | float | Learning rate. |
| `checkpoint.save_frequency` | int | Save checkpoint every N epochs. |
| `checkpoint.output_path` | string | Directory for checkpoint ZIPs. |
| `monitoring` | object\|null | Optional preview image generation during training. Set to null to disable. |
| `lora_layers.targets[]` | array | Each target: `module_path` (string), `rank` (int), optional `blocks` ({start, end} or {indices}). |

### Data Directory Format

mflux auto-discovers training data. Each image must have a matching `.txt` file:

```
data/
├── photo1.jpg
├── photo1.txt    # Contains caption: "a photo of ohwx"
├── photo2.png
├── photo2.txt    # Contains caption: "a photo of ohwx"
└── preview.txt   # Optional: used for monitoring preview prompts
```

**This replaces the plan's assumed `examples[]` array.** The `data` field is a path string, not a list.

### No `alpha` Field

mflux LoRA does NOT use an `alpha` parameter. Only `rank` controls the LoRA dimensionality. **Remove `alpha` from config.yaml search space.**

### Klein 4B Architecture

- **Model name:** `flux2-klein-base-4b`
- **Dual-stream blocks:** 5 (`transformer_blocks[0..4]`)
- **Single-stream blocks:** 20 (`single_transformer_blocks[0..19]`)
- **Attention heads:** 24, head dim: 128, inner dim: 3072
- **Joint attention dim:** 7680

### Klein 4B LoRA Target Module Paths

**Dual-stream blocks** (`transformer_blocks`): each has separate attention projections + feed-forward:
- `transformer_blocks.{block}.attn.to_q` (nn.Linear)
- `transformer_blocks.{block}.attn.to_k` (nn.Linear)
- `transformer_blocks.{block}.attn.to_v` (nn.Linear)
- `transformer_blocks.{block}.attn.to_out` (nn.Linear)
- `transformer_blocks.{block}.attn.add_q_proj` (nn.Linear, context attention)
- `transformer_blocks.{block}.attn.add_k_proj` (nn.Linear, context attention)
- `transformer_blocks.{block}.attn.add_v_proj` (nn.Linear, context attention)
- `transformer_blocks.{block}.attn.to_add_out` (nn.Linear, context attention)
- `transformer_blocks.{block}.ff.linear_in` (nn.Linear, feed-forward)
- `transformer_blocks.{block}.ff.linear_out` (nn.Linear, feed-forward)
- `transformer_blocks.{block}.ff_context.linear_in` (nn.Linear, context feed-forward)
- `transformer_blocks.{block}.ff_context.linear_out` (nn.Linear, context feed-forward)

Block range: `{ "start": 0, "end": 5 }`

**Single-stream blocks** (`single_transformer_blocks`): each has a FUSED QKV+MLP projection:
- `single_transformer_blocks.{block}.attn.to_qkv_mlp_proj` (nn.Linear, fused)
- `single_transformer_blocks.{block}.attn.to_out` (nn.Linear)

Block range: `{ "start": 0, "end": 20 }`

**Default LoRA config for Klein 4B (attention-only, dual-stream):**
```json
"lora_layers": {
  "targets": [
    { "module_path": "transformer_blocks.{block}.attn.to_q", "blocks": { "start": 0, "end": 5 }, "rank": 8 },
    { "module_path": "transformer_blocks.{block}.attn.to_k", "blocks": { "start": 0, "end": 5 }, "rank": 8 },
    { "module_path": "transformer_blocks.{block}.attn.to_v", "blocks": { "start": 0, "end": 5 }, "rank": 8 },
    { "module_path": "transformer_blocks.{block}.attn.to_out", "blocks": { "start": 0, "end": 5 }, "rank": 8 }
  ]
}
```

### Checkpoint Output

Trained adapters saved as ZIP files in `{checkpoint.output_path}/checkpoints/`:
- Filename: `{iterations:07d}_checkpoint.zip`
- Inside ZIP: `{iterations:07d}_adapter.safetensors`
- Must **extract the .safetensors from the ZIP** to use with `--lora-paths`

### Using Trained LoRA for Inference

```bash
# Extract adapter from latest checkpoint
unzip -o training/checkpoints/0000100_checkpoint.zip "0000100_adapter.safetensors" -d ./adapter/

# Generate with LoRA
mflux-generate \
  --model flux2-klein-base-4b \
  --lora-paths ./adapter/0000100_adapter.safetensors \
  --prompt "a photo of ohwx" \
  --seed 42 --steps 20 --width 1024 --height 1024 \
  --output eval.png
```

`--lora-paths` expects a **path to a .safetensors file** (not a directory).

---

## mlx_clip v0.2

### Installation

Not on PyPI. Install from GitHub:
```
pip install "mlx_clip @ git+https://github.com/harperreed/mlx_clip.git"
```

### API

```python
from mlx_clip import mlx_clip

# Initialize (auto-downloads weights on first use)
clip = mlx_clip(
    model_dir="~/.cache/autoresearch-lora/clip_model",
    hf_repo="openai/clip-vit-base-patch32"
)

# Embed an image — returns Python list of floats
embedding = clip.image_encoder("/path/to/image.png")

# Embed text — returns Python list of floats
text_emb = clip.text_encoder("a photo of a dog")
```

### Key Details

- Returns **Python lists**, not numpy arrays. Wrap with `np.array()` for cosine similarity.
- Auto-downloads and converts HuggingFace CLIP weights to MLX format on first use.
- Model stored in the provided `model_dir`.
- Default model: `openai/clip-vit-base-patch32` (embedding dim: 512).
- Uses MLX for inference (not PyTorch), runs natively on Apple Silicon.

---

## Impact on Plan

### Changes from original plan:

1. **Remove `alpha` from config.yaml** — mflux LoRA only uses `rank`
2. **`data` field is a directory path** — config_translator must set up a data directory with images + `.txt` caption files, not an `examples[]` array
3. **Checkpoint extraction** — train.py must extract `.safetensors` from checkpoint ZIP before passing to `mflux-generate --lora-paths`
4. **Default LoRA targets** — use Klein 4B-specific module paths (not z-image paths)
5. **`mflux-generate`** — not `mflux-generate-flux2` (the plan had wrong CLI name)
6. **mlx_clip install from git** — pyproject.toml must use git dependency
7. **`steps` in training config** — this is the denoising steps, not training iterations. Training iterations are controlled by `num_epochs × dataset_size / batch_size`

### Pinned versions:
- mflux: `==0.16.9`
- mlx_clip: `@ git+https://github.com/harperreed/mlx_clip.git@f56e3ecc72c74c68b6b50eb6f50c3f22fc23fe2c`

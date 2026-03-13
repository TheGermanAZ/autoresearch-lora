"""Scoring module for autoresearch-lora.

Provides CLIP image-image similarity (cosine, centroid, nearest-neighbor),
score aggregation, CLIP image embedding via mlx_clip, and optional VLM
judge scoring via Claude vision API.
"""

import base64
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Mean of embedding vectors."""
    return np.mean(embeddings, axis=0)


def score_against_centroid(eval_embedding: np.ndarray, centroid: np.ndarray) -> float:
    """Cosine similarity of one eval image against the reference centroid."""
    return cosine_similarity(eval_embedding, centroid)


def score_nearest_neighbor(
    eval_embedding: np.ndarray, ref_embeddings: np.ndarray
) -> float:
    """Max cosine similarity of eval image against any reference image."""
    sims = [cosine_similarity(eval_embedding, ref) for ref in ref_embeddings]
    return max(sims)


def aggregate_scores(
    per_prompt_centroid_sims: dict[int, list[float]],
    per_prompt_nn_sims: dict[int, list[float]],
    neg_sims: list[float],
    num_prompts: int,
) -> dict:
    """Aggregate per-image scores into experiment-level metrics.

    Args:
        per_prompt_centroid_sims: {prompt_index: [sim, ...]} for trigger prompts
        per_prompt_nn_sims: {prompt_index: [sim, ...]} for trigger prompts
        neg_sims: centroid sims for negative control images
        num_prompts: number of trigger prompts (excluding negative)
    """
    all_centroid = []
    all_nn = []
    prompt_scores = []
    for i in range(num_prompts):
        sims = per_prompt_centroid_sims.get(i, [])
        all_centroid.extend(sims)
        all_nn.extend(per_prompt_nn_sims.get(i, []))
        prompt_scores.append(float(np.mean(sims)) if sims else 0.0)

    return {
        "clip_sim_centroid": float(np.mean(all_centroid)) if all_centroid else 0.0,
        "clip_sim_nn": float(np.mean(all_nn)) if all_nn else 0.0,
        "prompt_scores": prompt_scores,
        "score_stddev": float(np.std(all_centroid)) if all_centroid else 0.0,
        "neg_control": float(np.mean(neg_sims)) if neg_sims else 0.0,
    }


# --- CLIP embedding (mlx_clip) ---

CLIP_MODEL_DIR = str(Path.home() / ".cache" / "autoresearch-lora" / "clip_model")
_clip_instance = None


def load_clip():
    """Load mlx_clip model (cached singleton)."""
    global _clip_instance
    if _clip_instance is None:
        from mlx_clip import mlx_clip

        _clip_instance = mlx_clip(
            model_dir=CLIP_MODEL_DIR,
            hf_repo="openai/clip-vit-base-patch32",
        )
    return _clip_instance


def embed_image(image_path: Path) -> np.ndarray:
    """Compute CLIP embedding for a single image. Returns numpy array."""
    clip = load_clip()
    embedding_list = clip.image_encoder(str(image_path))
    return np.array(embedding_list, dtype=np.float32)


# --- VLM Judge (Gemini 1.5 Pro via OpenRouter) ---

VLM_MODEL = "google/gemini-pro-1.5"
VLM_MAX_TOKENS = 60
VLM_TIMEOUT = 45  # seconds (OpenRouter can be slower)
VLM_API_URL = "https://openrouter.ai/api/v1/chat/completions"

VLM_RUBRIC = """\
Rate this AI-generated image on a scale of 1-10 for each criterion:

1. **Prompt adherence**: How well does the image match the generation prompt?
2. **Technical quality**: Is the image free of artifacts, distortions, or incoherent elements?
3. **Aesthetic appeal**: Is the image visually appealing and cohesive?

The generation prompt was: "{prompt}"

Respond with ONLY three numbers separated by commas, nothing else. Example: 7,8,6"""


def vlm_judge(image_path: Path, prompt: str, api_key: str | None = None) -> dict:
    """Score a single image using Gemini 1.5 Pro via OpenRouter.

    Returns dict with prompt_adherence, technical, aesthetic (each 0.0-1.0)
    and vlm_avg (arithmetic mean). Returns all zeros on failure or missing key.
    """
    zero = {"prompt_adherence": 0.0, "technical": 0.0, "aesthetic": 0.0, "vlm_avg": 0.0}

    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return zero

    # Read and base64-encode the image
    image_data = Path(image_path).read_bytes()
    b64 = base64.standard_b64encode(image_data).decode("ascii")
    suffix = Path(image_path).suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"

    rubric_text = VLM_RUBRIC.replace("{prompt}", prompt)

    # OpenRouter uses OpenAI-compatible chat completions format
    body = json.dumps({
        "model": VLM_MODEL,
        "max_tokens": VLM_MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                },
                {"type": "text", "text": rubric_text},
            ],
        }],
    }).encode()

    req = urllib.request.Request(
        VLM_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=VLM_TIMEOUT) as resp:
            result = json.loads(resp.read())
        text = result["choices"][0]["message"]["content"].strip()
        nums = re.findall(r"(\d+)", text)
        if len(nums) >= 3:
            adherence = min(float(nums[0]) / 10.0, 1.0)
            technical = min(float(nums[1]) / 10.0, 1.0)
            aesthetic = min(float(nums[2]) / 10.0, 1.0)
            vlm_avg = (adherence + technical + aesthetic) / 3.0
            return {
                "prompt_adherence": adherence,
                "technical": technical,
                "aesthetic": aesthetic,
                "vlm_avg": vlm_avg,
            }
    except Exception as e:
        print(f"  VLM error for {Path(image_path).name}: {e}")

    return zero


def vlm_judge_batch(
    image_prompt_pairs: list[tuple[Path, str]],
    api_key: str | None = None,
) -> dict:
    """Score multiple images and return aggregated VLM scores.

    Args:
        image_prompt_pairs: [(image_path, prompt_text), ...]

    Returns dict with mean scores across all images.
    """
    if not api_key:
        api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"vlm_avg": 0.0, "vlm_adherence": 0.0, "vlm_technical": 0.0, "vlm_aesthetic": 0.0}

    all_scores = []
    for img_path, prompt in image_prompt_pairs:
        s = vlm_judge(img_path, prompt, api_key)
        if s["vlm_avg"] > 0:
            all_scores.append(s)

    if not all_scores:
        return {"vlm_avg": 0.0, "vlm_adherence": 0.0, "vlm_technical": 0.0, "vlm_aesthetic": 0.0}

    return {
        "vlm_avg": float(np.mean([s["vlm_avg"] for s in all_scores])),
        "vlm_adherence": float(np.mean([s["prompt_adherence"] for s in all_scores])),
        "vlm_technical": float(np.mean([s["technical"] for s in all_scores])),
        "vlm_aesthetic": float(np.mean([s["aesthetic"] for s in all_scores])),
    }

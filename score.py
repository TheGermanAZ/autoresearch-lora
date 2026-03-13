"""CLIP scoring module for autoresearch-lora.

Provides cosine similarity, centroid scoring, nearest-neighbor scoring,
score aggregation, and CLIP image embedding via mlx_clip.
"""

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

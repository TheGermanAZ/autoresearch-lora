"""CLIP scoring module for autoresearch-lora.

Provides cosine similarity, centroid scoring, nearest-neighbor scoring,
score aggregation, and CLIP image embedding via mlx_clip.
"""

from pathlib import Path

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


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

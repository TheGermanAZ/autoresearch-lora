import numpy as np


def test_cosine_similarity_identical():
    from score import cosine_similarity

    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    from score import cosine_similarity

    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_similarity_opposite():
    from score import cosine_similarity

    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6


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
    assert abs(sim - 1.0) < 1e-6


def test_score_nearest_neighbor_picks_closest():
    from score import score_nearest_neighbor

    eval_embedding = np.array([0.9, 0.1])
    ref_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = score_nearest_neighbor(eval_embedding, ref_embeddings)
    # Should pick the first ref (closer to [0.9, 0.1])
    from score import cosine_similarity

    expected = cosine_similarity(eval_embedding, ref_embeddings[0])
    assert abs(sim - expected) < 1e-6


def test_aggregate_scores():
    from score import aggregate_scores

    centroid_sims = [0.8] * 20
    nn_sims = [0.75] * 20
    neg_sims = [0.3] * 4
    result = aggregate_scores(
        centroid_sims, nn_sims, neg_sims, num_prompts=5, seeds_per_prompt=4
    )
    assert abs(result["clip_sim_centroid"] - 0.8) < 1e-6
    assert abs(result["clip_sim_nn"] - 0.75) < 1e-6
    assert abs(result["neg_control"] - 0.3) < 1e-6
    assert len(result["prompt_scores"]) == 5
    assert "score_stddev" in result


def test_aggregate_scores_varying():
    from score import aggregate_scores

    # 5 prompts × 4 seeds, with different scores per prompt
    centroid_sims = [0.9] * 4 + [0.8] * 4 + [0.7] * 4 + [0.85] * 4 + [0.75] * 4
    nn_sims = [0.85] * 20
    neg_sims = [0.35] * 4
    result = aggregate_scores(
        centroid_sims, nn_sims, neg_sims, num_prompts=5, seeds_per_prompt=4
    )
    assert abs(result["prompt_scores"][0] - 0.9) < 1e-6
    assert abs(result["prompt_scores"][1] - 0.8) < 1e-6
    assert abs(result["prompt_scores"][2] - 0.7) < 1e-6
    assert result["score_stddev"] > 0

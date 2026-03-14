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

    per_prompt_centroid = {i: [0.8] * 4 for i in range(5)}
    per_prompt_nn = {i: [0.75] * 4 for i in range(5)}
    neg_sims = [0.3] * 4
    result = aggregate_scores(
        per_prompt_centroid, per_prompt_nn, neg_sims, num_prompts=5
    )
    assert abs(result["clip_sim_centroid"] - 0.8) < 1e-6
    assert abs(result["clip_sim_nn"] - 0.75) < 1e-6
    assert abs(result["neg_control"] - 0.3) < 1e-6
    assert len(result["prompt_scores"]) == 5
    assert "score_stddev" in result


def test_aggregate_scores_varying():
    from score import aggregate_scores

    # 5 prompts × 4 seeds, with different scores per prompt
    per_prompt_centroid = {
        0: [0.9] * 4,
        1: [0.8] * 4,
        2: [0.7] * 4,
        3: [0.85] * 4,
        4: [0.75] * 4,
    }
    per_prompt_nn = {i: [0.85] * 4 for i in range(5)}
    neg_sims = [0.35] * 4
    result = aggregate_scores(
        per_prompt_centroid, per_prompt_nn, neg_sims, num_prompts=5
    )
    assert abs(result["prompt_scores"][0] - 0.9) < 1e-6
    assert abs(result["prompt_scores"][1] - 0.8) < 1e-6
    assert abs(result["prompt_scores"][2] - 0.7) < 1e-6
    assert result["score_stddev"] > 0


def test_aggregate_scores_partial_generation():
    """If some images fail to generate, scores should still be correct."""
    from score import aggregate_scores

    # Prompt 2 only got 2 of 4 seeds
    per_prompt_centroid = {
        0: [0.9] * 4,
        1: [0.8] * 4,
        2: [0.7, 0.7],
        3: [0.85] * 4,
        4: [0.75] * 4,
    }
    per_prompt_nn = {
        0: [0.85] * 4,
        1: [0.85] * 4,
        2: [0.80, 0.80],
        3: [0.85] * 4,
        4: [0.85] * 4,
    }
    result = aggregate_scores(
        per_prompt_centroid, per_prompt_nn, [0.3], num_prompts=5
    )
    assert abs(result["prompt_scores"][2] - 0.7) < 1e-6
    assert len(result["prompt_scores"]) == 5


def test_aggregate_scores_empty_neg():
    """Empty neg_sims should return 0.0 for neg_control."""
    from score import aggregate_scores

    per_prompt_centroid = {0: [0.8] * 4}
    per_prompt_nn = {0: [0.75] * 4}
    result = aggregate_scores(
        per_prompt_centroid, per_prompt_nn, [], num_prompts=1
    )
    assert result["neg_control"] == 0.0


def test_cosine_similarity_zero_norm():
    """Zero-norm vector should return 0.0, not crash."""
    from score import cosine_similarity

    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    assert cosine_similarity(a, b) == 0.0


def test_score_nearest_neighbor_empty_refs():
    """Empty ref_embeddings should return 0.0, not crash."""
    from score import score_nearest_neighbor

    eval_embedding = np.array([1.0, 0.0])
    ref_embeddings = np.array([]).reshape(0, 2)
    assert score_nearest_neighbor(eval_embedding, ref_embeddings) == 0.0


def test_aggregate_scores_fully_empty():
    """Completely empty inputs should return all zeros."""
    from score import aggregate_scores

    result = aggregate_scores({}, {}, [], num_prompts=0)
    assert result["clip_sim_centroid"] == 0.0
    assert result["clip_sim_nn"] == 0.0
    assert result["neg_control"] == 0.0
    assert result["prompt_scores"] == []
    assert result["score_stddev"] == 0.0


def test_vlm_judge_parses_simple_csv(tmp_path, monkeypatch):
    """VLM returns clean '7,8,6'."""
    import json
    import urllib.request
    from score import vlm_judge

    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    body = json.dumps({"choices": [{"message": {"content": "7,8,6"}}]}).encode()

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: FakeResp())
    result = vlm_judge(img, "a photo of ohwx", api_key="fake-key")
    assert abs(result["prompt_adherence"] - 0.7) < 1e-6
    assert abs(result["technical"] - 0.8) < 1e-6
    assert abs(result["aesthetic"] - 0.6) < 1e-6
    assert abs(result["vlm_avg"] - 0.7) < 1e-6


def test_vlm_judge_parses_slash_ten_format(tmp_path, monkeypatch):
    """VLM returns '7/10, 8/10, 6/10' — the format the regex fix targets."""
    import json
    import urllib.request
    from score import vlm_judge

    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    body = json.dumps({"choices": [{"message": {"content": "7/10, 8/10, 6/10"}}]}).encode()

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: FakeResp())
    result = vlm_judge(img, "test prompt", api_key="fake-key")
    assert abs(result["prompt_adherence"] - 0.7) < 1e-6
    assert abs(result["technical"] - 0.8) < 1e-6
    assert abs(result["aesthetic"] - 0.6) < 1e-6


def test_vlm_judge_parses_verbose_response(tmp_path, monkeypatch):
    """VLM returns prose with numbers — regex extracts first number per part."""
    import json
    import urllib.request
    from score import vlm_judge

    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    body = json.dumps({"choices": [{"message": {"content": "Prompt adherence: 9, Technical quality: 7, Aesthetic appeal: 8"}}]}).encode()

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: FakeResp())
    result = vlm_judge(img, "test", api_key="fake-key")
    assert abs(result["prompt_adherence"] - 0.9) < 1e-6
    assert abs(result["technical"] - 0.7) < 1e-6
    assert abs(result["aesthetic"] - 0.8) < 1e-6


def test_vlm_judge_garbage_response_returns_zero(tmp_path, monkeypatch):
    """VLM returns nonsense with <3 extractable numbers -> zeros."""
    import json
    import urllib.request
    from score import vlm_judge

    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    body = json.dumps({"choices": [{"message": {"content": "I cannot rate this image."}}]}).encode()

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: FakeResp())
    result = vlm_judge(img, "test", api_key="fake-key")
    assert result["vlm_avg"] == 0.0


def test_vlm_judge_clamps_above_10(tmp_path, monkeypatch):
    """Scores > 10 should clamp to 1.0."""
    import json
    import urllib.request
    from score import vlm_judge

    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    body = json.dumps({"choices": [{"message": {"content": "15,20,30"}}]}).encode()

    class FakeResp:
        def read(self):
            return body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: FakeResp())
    result = vlm_judge(img, "test", api_key="fake-key")
    assert result["prompt_adherence"] == 1.0
    assert result["technical"] == 1.0
    assert result["aesthetic"] == 1.0


def test_vlm_judge_network_error_returns_zero(tmp_path, monkeypatch):
    """Network failure should return zeros."""
    import urllib.error
    import urllib.request
    from score import vlm_judge

    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("timeout")))
    result = vlm_judge(img, "test", api_key="fake-key")
    assert result["vlm_avg"] == 0.0


def test_vlm_judge_batch_filters_zero_scores(tmp_path, monkeypatch):
    """vlm_judge_batch should exclude zero-scored images from the mean."""
    import json
    import urllib.request
    from score import vlm_judge_batch

    img1 = tmp_path / "good.png"
    img1.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    img2 = tmp_path / "bad.png"
    img2.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    call_count = {"n": 0}

    def fake_urlopen(req, timeout=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            data = json.dumps({"choices": [{"message": {"content": "8,9,7"}}]}).encode()
        else:
            data = json.dumps({"choices": [{"message": {"content": "nonsense"}}]}).encode()

        class FakeResp:
            def read(self):
                return data
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
        return FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = vlm_judge_batch([(img1, "p1"), (img2, "p2")], api_key="fake-key")
    assert abs(result["vlm_avg"] - 0.8) < 1e-6


def test_vlm_judge_no_api_key():
    """Without API key, vlm_judge returns zeros."""
    import os
    from pathlib import Path

    from score import vlm_judge

    # Ensure no key in env
    old = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        result = vlm_judge(Path("/nonexistent.png"), "test prompt", api_key=None)
        assert result["vlm_avg"] == 0.0
        assert result["prompt_adherence"] == 0.0
        assert result["technical"] == 0.0
        assert result["aesthetic"] == 0.0
    finally:
        if old:
            os.environ["OPENROUTER_API_KEY"] = old


def test_vlm_judge_batch_no_api_key():
    """Without API key, vlm_judge_batch returns zeros."""
    import os
    from pathlib import Path

    from score import vlm_judge_batch

    old = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        result = vlm_judge_batch([(Path("/fake.png"), "test")], api_key=None)
        assert result["vlm_avg"] == 0.0
    finally:
        if old:
            os.environ["OPENROUTER_API_KEY"] = old

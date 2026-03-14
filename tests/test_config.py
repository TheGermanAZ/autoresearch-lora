import json
import tempfile
from pathlib import Path

import pytest


def test_load_config():
    from config_translator import load_config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("rank: 16\nlr: 0.0003\nsteps: 500\n")
        f.flush()
        config = load_config(Path(f.name))
    assert config["rank"] == 16
    assert config["lr"] == 0.0003
    assert config["steps"] == 500


def test_load_config_invalid_yaml():
    from config_translator import ConfigError, load_config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(": invalid: yaml: [")
        f.flush()
        with pytest.raises(ConfigError):
            load_config(Path(f.name))


def test_to_mflux_json_basic():
    from config_translator import to_mflux_json

    config = {
        "rank": 16,
        "lr": 3e-4,
        "batch_size": 1,
        "steps": 500,
        "num_epochs": 1,
        "quantize": 4,
        "guidance": 4.0,
        "trigger_word": "ohwx",
        "caption_template": "a photo of {trigger}",
    }
    data_dir = Path("/tmp/test-data")
    output_dir = Path("/tmp/test-output")
    mflux_config = to_mflux_json(config, data_dir, output_dir)

    assert mflux_config["model"] == "flux2-klein-base-4b"
    assert mflux_config["steps"] == 500
    assert mflux_config["guidance"] == 4.0
    assert mflux_config["quantize"] == 4
    assert mflux_config["data"] == str(data_dir)
    assert mflux_config["optimizer"]["name"] == "AdamW"
    assert mflux_config["optimizer"]["learning_rate"] == 3e-4
    assert mflux_config["training_loop"]["num_epochs"] == 1
    assert mflux_config["training_loop"]["batch_size"] == 1
    assert mflux_config["training_loop"]["timestep_low"] == 0
    assert mflux_config["training_loop"]["timestep_high"] is None
    assert isinstance(mflux_config["lora_layers"]["targets"], list)
    assert len(mflux_config["lora_layers"]["targets"]) > 0


def test_to_mflux_json_rank_propagates():
    from config_translator import to_mflux_json

    config = {
        "rank": 32,
        "lr": 1e-4,
        "batch_size": 2,
        "steps": 9,
        "num_epochs": 50,
        "quantize": None,
        "guidance": 0.0,
        "trigger_word": "sks",
        "caption_template": "a photo of {trigger}",
    }
    mflux_config = to_mflux_json(config, Path("/tmp/data"), Path("/tmp/out"))
    for target in mflux_config["lora_layers"]["targets"]:
        assert target["rank"] == 32


def test_to_mflux_json_no_quantize():
    from config_translator import to_mflux_json

    config = {
        "rank": 8,
        "lr": 3e-4,
        "batch_size": 1,
        "steps": 9,
        "num_epochs": 1,
        "quantize": None,
        "guidance": 4.0,
        "trigger_word": "ohwx",
        "caption_template": "a photo of {trigger}",
    }
    mflux_config = to_mflux_json(config, Path("/tmp/data"), Path("/tmp/out"))
    assert mflux_config["quantize"] is None


def test_write_mflux_json():
    from config_translator import to_mflux_json, write_mflux_json

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
    mflux_config = to_mflux_json(config, Path("/tmp/data"), Path("/tmp/out"))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_path = Path(f.name)

    write_mflux_json(mflux_config, out_path)
    loaded = json.loads(out_path.read_text())
    assert loaded["model"] == "flux2-klein-base-4b"
    assert loaded["steps"] == 9


def test_prepare_data_dir(tmp_path):
    from config_translator import prepare_data_dir

    # Create source images
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for i in range(3):
        (source_dir / f"img{i}.jpg").write_bytes(b"fake-image-data")

    config = {
        "trigger_word": "ohwx",
        "caption_template": "a photo of {trigger}",
    }
    data_dir = tmp_path / "data"
    prepare_data_dir(config, source_dir, data_dir)

    # Should have images + matching .txt files
    images = sorted(data_dir.glob("*.jpg"))
    txts = sorted(data_dir.glob("*.txt"))
    assert len(images) == 3
    assert len(txts) == 3

    # Check captions
    for txt in txts:
        assert txt.read_text().strip() == "a photo of ohwx"


def test_to_mflux_json_all_defaults():
    """Empty config should produce valid JSON with all defaults."""
    from config_translator import to_mflux_json

    result = to_mflux_json({}, Path("/tmp/data"), Path("/tmp/out"))
    assert result["model"] == "flux2-klein-base-4b"
    assert result["steps"] == 9
    assert result["guidance"] == 4.0
    assert result["quantize"] is None
    assert result["optimizer"]["learning_rate"] == 3e-4
    assert result["training_loop"]["num_epochs"] == 1
    assert result["training_loop"]["batch_size"] == 1
    for target in result["lora_layers"]["targets"]:
        assert target["rank"] == 8


def test_to_mflux_json_rank_does_not_mutate_defaults():
    """Successive calls must not leak rank values between calls."""
    from config_translator import KLEIN_4B_DEFAULT_TARGETS, to_mflux_json

    to_mflux_json({"rank": 32}, Path("/d"), Path("/o"))
    result2 = to_mflux_json({"rank": 4}, Path("/d"), Path("/o"))

    for target in result2["lora_layers"]["targets"]:
        assert target["rank"] == 4

    # Original defaults must be untouched
    for t in KLEIN_4B_DEFAULT_TARGETS:
        assert "rank" not in t


def test_to_mflux_json_quantize_string_coercion():
    """YAML may parse '4' as string; should coerce to int."""
    from config_translator import to_mflux_json

    result = to_mflux_json({"quantize": "4"}, Path("/d"), Path("/o"))
    assert result["quantize"] == 4
    assert isinstance(result["quantize"], int)


def test_to_mflux_json_invalid_quantize():
    """Invalid quantize value should raise ConfigError."""
    from config_translator import ConfigError, to_mflux_json

    try:
        to_mflux_json({"quantize": 7}, Path("/d"), Path("/o"))
        assert False, "Should have raised ConfigError"
    except ConfigError:
        pass


def test_prepare_data_dir_cleans_stale_files(tmp_path):
    """Stale files from a previous run must be deleted."""
    from config_translator import prepare_data_dir

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "img0.jpg").write_bytes(b"fake")
    (source_dir / "img1.jpg").write_bytes(b"fake")

    config = {"trigger_word": "ohwx", "caption_template": "a photo of {trigger}"}
    data_dir = tmp_path / "data"

    # First run: 2 images
    prepare_data_dir(config, source_dir, data_dir)
    assert len(list(data_dir.glob("*.jpg"))) == 2

    # Remove one source image
    (source_dir / "img1.jpg").unlink()
    prepare_data_dir(config, source_dir, data_dir)

    # Must have exactly 1 image, not 2
    assert len(list(data_dir.glob("*.jpg"))) == 1
    assert len(list(data_dir.glob("*.txt"))) == 1


def test_prepare_data_dir_skips_non_images(tmp_path):
    """Non-image files should not be copied."""
    from config_translator import prepare_data_dir

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "photo.png").write_bytes(b"fake")
    (source_dir / "notes.txt").write_text("not an image")
    (source_dir / ".DS_Store").write_bytes(b"\x00")

    config = {"trigger_word": "sks", "caption_template": "{trigger} portrait"}
    data_dir = tmp_path / "data"
    prepare_data_dir(config, source_dir, data_dir)

    assert len(list(data_dir.glob("*.png"))) == 1
    txts = list(data_dir.glob("*.txt"))
    assert len(txts) == 1
    assert txts[0].read_text().strip() == "sks portrait"


def test_to_mflux_json_checkpoint_save_frequency():
    """save_frequency must equal num_epochs to ensure checkpoint exists."""
    from config_translator import to_mflux_json

    for epochs in [1, 5, 50]:
        result = to_mflux_json({"num_epochs": epochs}, Path("/d"), Path("/o"))
        assert result["checkpoint"]["save_frequency"] == epochs


def test_load_config_non_dict():
    """YAML that parses to a non-dict should raise ConfigError."""
    from config_translator import ConfigError, load_config

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("- item1\n- item2\n")
        f.flush()
        with pytest.raises(ConfigError, match="must be a YAML mapping"):
            load_config(Path(f.name))

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

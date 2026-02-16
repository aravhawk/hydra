import json
import pytest
from pathlib import Path
from hydra.types import Defense
from hydra.export import DefenseExporter


@pytest.fixture
def exporter():
    return DefenseExporter()


@pytest.fixture
def sample_defenses():
    return [
        Defense(
            id="d1",
            name="block_injection",
            trigger_attack_id="a1",
            colang_code="define flow block_injection\n  user ...\n  bot refuse",
            config_yaml="",
        ),
        Defense(
            id="d2",
            name="block_jailbreak",
            trigger_attack_id="a2",
            colang_code="define flow block_jailbreak\n  user ...\n  bot refuse",
            config_yaml="",
        ),
    ]


def test_export_creates_files(tmp_path, exporter, sample_defenses):
    out = exporter.export(sample_defenses, tmp_path / "defenses", "nemotron-3-nano",
                          "http://localhost:11434/v1")
    assert out is not None
    assert (out / "config.yml").exists()
    assert (out / "defenses.co").exists()
    assert (out / "manifest.json").exists()


def test_config_content(tmp_path, exporter, sample_defenses):
    out = exporter.export(sample_defenses, tmp_path / "defenses", "nemotron-3-nano",
                          "http://localhost:11434/v1")
    config = (out / "config.yml").read_text()
    assert "nemotron-3-nano" in config
    assert "block_injection" in config
    assert "block_jailbreak" in config


def test_colang_content(tmp_path, exporter, sample_defenses):
    out = exporter.export(sample_defenses, tmp_path / "defenses", "nemotron-3-nano",
                          "http://localhost:11434/v1")
    colang = (out / "defenses.co").read_text()
    assert "define flow block_injection" in colang
    assert "define flow block_jailbreak" in colang
    assert "# Defense: block_injection" in colang


def test_manifest_structure(tmp_path, exporter, sample_defenses):
    out = exporter.export(sample_defenses, tmp_path / "defenses", "nemotron-3-nano",
                          "http://localhost:11434/v1")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["generated_by"] == "hydra"
    assert manifest["model"] == "nemotron-3-nano"
    assert manifest["defense_count"] == 2
    assert len(manifest["defenses"]) == 2
    assert manifest["defenses"][0]["name"] == "block_injection"


def test_empty_defenses_returns_none(tmp_path, exporter):
    result = exporter.export([], tmp_path / "defenses", "nemotron-3-nano",
                             "http://localhost:11434/v1")
    assert result is None
    assert not (tmp_path / "defenses").exists()


def test_deep_directory_creation(tmp_path, exporter, sample_defenses):
    deep_path = tmp_path / "a" / "b" / "c" / "defenses"
    out = exporter.export(sample_defenses, deep_path, "nemotron-3-nano",
                          "http://localhost:11434/v1")
    assert out is not None
    assert out.exists()
    assert (out / "config.yml").exists()

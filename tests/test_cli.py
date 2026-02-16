import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from typer.testing import CliRunner
from hydra.cli import app
from hydra.types import Defense

runner = CliRunner()


def _mock_engine_scan():
    from hydra.types import ScanResult
    return ScanResult(
        target_name="nemotron-3-nano",
        rounds=2,
        total_attacks=6,
        bypassed=3,
        blocked=3,
        vulnerability_score=0.5,
        duration_seconds=1.0,
    )


def _mock_engine_scan_with_defenses():
    from hydra.types import ScanResult
    return ScanResult(
        target_name="nemotron-3-nano",
        rounds=2,
        total_attacks=6,
        bypassed=3,
        blocked=3,
        vulnerability_score=0.5,
        duration_seconds=1.0,
        defenses=[
            Defense(
                id="d1",
                name="block_injection",
                trigger_attack_id="a1",
                colang_code="define flow block_injection\n  user ...\n  bot refuse",
                config_yaml="",
            ),
        ],
    )


def _mock_config():
    mock_config = MagicMock()
    mock_config.ollama_base_url = "http://localhost:11434/v1"
    mock_config.llm_model = "nemotron-3-nano"
    mock_config.guardrails_enabled = False
    mock_config.guardrails_model = ""
    mock_config.hardening_output_dir = "./hardened_model"
    mock_config.hardening_lora_rank = 16
    mock_config.hardening_epochs = 3
    mock_config.hardening_learning_rate = 2e-4
    mock_config.hardening_batch_size = 2
    mock_config.hardening_quantization = "q4_k_m"
    return mock_config


def test_scan_runs():
    with patch("hydra.cli.HydraConfig") as mock_config_cls, \
         patch("hydra.cli.HydraEngine") as mock_engine_cls:
        mock_config_cls.return_value = _mock_config()

        mock_engine = MagicMock()
        mock_engine.run_scan = AsyncMock(return_value=_mock_engine_scan())
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["--rounds", "1"])
        assert result.exit_code == 0


def test_scan_with_model():
    with patch("hydra.cli.HydraConfig") as mock_config_cls, \
         patch("hydra.cli.HydraEngine") as mock_engine_cls:
        mock_config_cls.return_value = _mock_config()

        mock_engine = MagicMock()
        mock_engine.run_scan = AsyncMock(return_value=_mock_engine_scan())
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["--model", "llama3", "--rounds", "1"])
        assert result.exit_code == 0


def test_scan_with_guardrails_flag():
    with patch("hydra.cli.HydraConfig") as mock_config_cls, \
         patch("hydra.cli.HydraEngine") as mock_engine_cls:
        mock_config_cls.return_value = _mock_config()

        mock_engine = MagicMock()
        mock_engine.run_scan = AsyncMock(return_value=_mock_engine_scan())
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["--rounds", "1", "--guardrails"])
        assert result.exit_code == 0


def test_export_defenses_flag(tmp_path):
    with patch("hydra.cli.HydraConfig") as mock_config_cls, \
         patch("hydra.cli.HydraEngine") as mock_engine_cls:
        mock_config_cls.return_value = _mock_config()

        mock_engine = MagicMock()
        mock_engine.run_scan = AsyncMock(return_value=_mock_engine_scan_with_defenses())
        mock_engine_cls.return_value = mock_engine

        export_dir = str(tmp_path / "exported")
        result = runner.invoke(app, ["--rounds", "1", "--export-defenses", export_dir])
        assert result.exit_code == 0
        assert "Defenses Exported" in result.output


def test_export_defenses_no_defenses():
    with patch("hydra.cli.HydraConfig") as mock_config_cls, \
         patch("hydra.cli.HydraEngine") as mock_engine_cls:
        mock_config_cls.return_value = _mock_config()

        mock_engine = MagicMock()
        mock_engine.run_scan = AsyncMock(return_value=_mock_engine_scan())
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["--rounds", "1", "--export-defenses", "/tmp/test"])
        assert result.exit_code == 0
        assert "nothing to export" in result.output.lower()


def test_harden_flag_without_deps():
    with patch("hydra.cli.HydraConfig") as mock_config_cls, \
         patch("hydra.cli.HydraEngine") as mock_engine_cls, \
         patch("hydra.hardening.FINETUNE_AVAILABLE", False):
        mock_config_cls.return_value = _mock_config()

        mock_engine = MagicMock()
        mock_engine.run_scan = AsyncMock(return_value=_mock_engine_scan())
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["--rounds", "1", "--harden"])
        assert result.exit_code == 0
        assert "not installed" in result.output.lower()


def test_verify_without_harden():
    with patch("hydra.cli.HydraConfig") as mock_config_cls, \
         patch("hydra.cli.HydraEngine") as mock_engine_cls:
        mock_config_cls.return_value = _mock_config()

        mock_engine = MagicMock()
        mock_engine.run_scan = AsyncMock(return_value=_mock_engine_scan())
        mock_engine_cls.return_value = mock_engine

        result = runner.invoke(app, ["--rounds", "1", "--verify"])
        assert result.exit_code == 0
        assert "--verify requires --harden" in result.output

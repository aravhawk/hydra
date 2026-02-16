"""Defense Export — saves generated Colang defenses to disk as a deployable NeMo Guardrails config."""

import json
from datetime import datetime, timezone
from pathlib import Path

from hydra.types import Defense


class DefenseExporter:
    """Writes defenses to disk as a deployable NeMo Guardrails config directory."""

    def export(self, defenses: list[Defense], output_dir: str | Path,
               model: str, base_url: str) -> Path | None:
        """Export defenses to a directory with config.yml, defenses.co, and manifest.json.

        Returns the output path, or None if no defenses to export.
        """
        if not defenses:
            return None

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        flow_names = self._extract_flow_names(defenses)
        config_yaml = self._build_config_yaml(flow_names, model, base_url)
        colang = self._build_colang(defenses)
        manifest = self._build_manifest(defenses, model)

        (out / "config.yml").write_text(config_yaml)
        (out / "defenses.co").write_text(colang)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

        return out

    def _extract_flow_names(self, defenses: list[Defense]) -> list[str]:
        """Parse 'define flow' lines from Colang code to get flow names."""
        names = []
        for d in defenses:
            for line in d.colang_code.split("\n"):
                stripped = line.strip()
                if stripped.startswith("define flow "):
                    name = stripped[len("define flow "):].strip()
                    if name:
                        names.append(name)
        return names

    def _build_config_yaml(self, flow_names: list[str], model: str,
                           base_url: str) -> str:
        """Generate NeMo Guardrails config YAML."""
        flows_yaml = "\n".join(f"      - {name}" for name in flow_names)

        return (
            f"models:\n"
            f"  - type: main\n"
            f"    engine: openai\n"
            f"    model: {model}\n"
            f"    parameters:\n"
            f"      base_url: {base_url}\n"
            f"      api_key: ollama\n"
            f"rails:\n"
            f"  input:\n"
            f"    flows:\n"
            f"{flows_yaml}\n"
        )

    def _build_colang(self, defenses: list[Defense]) -> str:
        """Concatenate all defense Colang code with comments."""
        parts = []
        for d in defenses:
            parts.append(f"# Defense: {d.name} (trigger: {d.trigger_attack_id})")
            parts.append(d.colang_code)
            parts.append("")
        return "\n".join(parts)

    def _build_manifest(self, defenses: list[Defense], model: str) -> dict:
        """Build metadata manifest for the exported defenses."""
        return {
            "generated_by": "hydra",
            "model": model,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "defense_count": len(defenses),
            "defenses": [
                {
                    "id": d.id,
                    "name": d.name,
                    "trigger_attack_id": d.trigger_attack_id,
                }
                for d in defenses
            ],
        }

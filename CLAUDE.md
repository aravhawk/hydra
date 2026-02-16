# Hydra - AI Security Platform

## Overview
Open-source AI red teaming platform. Red team agents attack a target AI, blue team agents classify results and generate NeMo Guardrails Colang defenses, then a mutator evolves attacks to bypass those defenses.

**Core loop:** Attack -> Detect -> Defend -> Mutate -> Repeat.

## Architecture
- `hydra/config.py` — Pydantic settings (env prefix: `HYDRA_`)
- `hydra/types.py` — All Pydantic models (Attack, AttackResult, Defense, ScanResult, HardeningResult)
- `hydra/llm.py` — LLM provider (Ollama via openai SDK, `http://localhost:11434/v1`)
- `hydra/target.py` — Target interface (model being attacked)
- `hydra/engine.py` — Main scan loop (sequential, no concurrency)
- `hydra/guardrails.py` — NeMo Guardrails runtime integration (optional)
- `hydra/agents/` — Red team (Injector, Jailbreaker, Exfiltrator, Mutator) + Blue team (Classifier, Defender)
- `hydra/cli.py` — Typer CLI (`hydra scan`)
- `hydra/export.py` — Defense export (writes Colang defenses to disk)
- `hydra/hardening.py` — LoRA fine-tuning pipeline (optional, requires `.[hardening]`)

## LLM Backend
- All inference via **Ollama** running locally (default: `http://localhost:11434/v1`)
- Default model: `nemotron-3-nano` (NVIDIA Nemotron 3 Nano via Ollama)
- Uses `openai` Python SDK pointed at Ollama's OpenAI-compatible endpoint
- No API keys required

## NeMo Guardrails Integration
- Optional: enabled via `--guardrails` CLI flag or `HYDRA_GUARDRAILS_ENABLED=true`
- Defenses are deployed live into the guardrails runtime as they're generated
- Attacks are tested against guardrails before reaching the target
- Install with: `pip install -e ".[guardrails]"`

## Defense Export
- `hydra/export.py` — `DefenseExporter` writes Colang defenses to disk
- Generates `config.yml`, `defenses.co`, `manifest.json` in the output directory
- Triggered via `--export-defenses PATH` CLI flag

## Hardening (LoRA Fine-Tuning)
- `hydra/hardening.py` — Optional module (same pattern as `guardrails.py`)
- `FINETUNE_AVAILABLE` flag via try/except on `unsloth`, `trl`, `datasets`, `torch`
- `TrainingDataGenerator` — converts bypassed attacks into SFT refusal training data
- `ModelHardener` — LoRA fine-tune with unsloth, export GGUF, import into Ollama as `{model}-hardened`
- Model loading is now resilient: Nemotron defaults to 16-bit load for stability, and non-Nemotron models retry from 4-bit to 16-bit automatically if 4-bit init fails
- `run_hardening_pipeline()` — top-level orchestrator called from CLI
- Install deps: `pip install -e ".[hardening]"`
- Config fields in `HydraConfig`: `hardening_output_dir`, `hardening_lora_rank`, `hardening_epochs`, `hardening_learning_rate`, `hardening_batch_size`, `hardening_quantization`

## Commands
- `hydra scan` — Scan the default model
- `hydra scan --model <name> --rounds <n>` — Scan a specific model
- `hydra scan --guardrails` — Scan with live NeMo Guardrails defense testing
- `hydra scan --export-defenses PATH` — Export defenses to a deployable directory
- `hydra scan --harden` — Run LoRA fine-tuning pipeline after scan
- `hydra scan --harden --verify` — Harden and re-scan to measure improvement
- `pytest tests/` — Run all tests

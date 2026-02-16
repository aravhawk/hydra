# Hydra

**Open-source AI red teaming platform with automated remediation.**

> "Cut off one head, two grow back." Patch one vulnerability, Hydra mutates
> and generates new attacks that bypass the fix.

Hydra runs a continuous adversarial loop against any model on Ollama: red team agents attack, a classifier detects bypasses, a defender generates NeMo Guardrails Colang defenses, and a mutator evolves attacks to beat those defenses. After scanning, Hydra can export defenses to disk and LoRA fine-tune the target model to harden it against discovered vulnerabilities.

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- A model pulled: `ollama pull nemotron-3-nano`

## Quick Start

```bash
pip install -e .

# Scan the default model (nemotron-3-nano)
hydra

# Scan a specific model with custom rounds
hydra --model nemotron-3-nano --rounds 5

# Scan with live NeMo Guardrails defense testing
pip install -e ".[guardrails]"
hydra --rounds 3 --guardrails
```

## Defense Export

Export generated Colang defenses to disk as a deployable NeMo Guardrails config directory:

```bash
hydra --rounds 3 --export-defenses ./my_defenses
```

Creates a directory with:
- `config.yml` — NeMo Guardrails config with model/engine/flows
- `defenses.co` — Colang defense rules
- `manifest.json` — Defense metadata

## Model Hardening (LoRA Fine-Tuning)

Use bypassed attack results to LoRA fine-tune the target model, making it resistant to discovered vulnerabilities:

```bash
# Install hardening dependencies (requires NVIDIA GPU)
pip install -e ".[hardening]"

# Scan and harden — fine-tunes and exports to Ollama as {model}-hardened
hydra --rounds 3 --harden

# Scan, harden, and verify — re-scans hardened model to measure improvement
hydra --rounds 3 --harden --verify
```

After hardening, the new model is available in Ollama:
```bash
ollama list  # shows nemotron-3-nano-hardened
ollama run nemotron-3-nano-hardened
```

Notes:
- Nemotron-3-Nano hardening uses 16-bit loading for stability on recent transformers/bitsandbytes stacks.
- Plan for high-memory GPUs when hardening Nemotron-class models (30B params, ~60GB VRAM at 16-bit).

## How It Works

1. **Red team agents** generate attacks (prompt injection, jailbreaks, data exfiltration)
2. **Target AI** processes each attack
3. **Classifier** determines if the attack bypassed safety measures
4. **Defender** auto-generates NeMo Guardrails Colang defense rules
5. **Mutator** evolves successful attacks to bypass new defenses
6. **Repeat** — each round, attacks get more sophisticated
7. **Export** (optional) — save defenses to disk as deployable guardrails config
8. **Harden** (optional) — LoRA fine-tune the target model with refusal training data from bypassed attacks

With `--guardrails`, defenses are deployed into a live NeMo Guardrails runtime. Attacks are tested against the guardrails before reaching the target — blocked attacks never touch the model.

## NVIDIA Ecosystem Integration

- **Nemotron 3 Nano** via Ollama — powers all agents locally on GPU
- **NeMo Guardrails Colang** — defense output format + live runtime testing
- **NemoGuard Safety Taxonomy** — 23 safety categories used for classification
- **Garak-inspired** — attack techniques drawn from NVIDIA's probe library
- **Unsloth + LoRA** — efficient fine-tuning for model hardening

## License

Apache 2.0

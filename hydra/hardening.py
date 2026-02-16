"""LoRA fine-tuning pipeline for model hardening (optional).

Requires: pip install -e ".[hardening]"
"""

import glob
import logging
import subprocess
from pathlib import Path

from rich.console import Console

from hydra.llm import LLMProvider
from hydra.types import ScanResult, HardeningResult

logger = logging.getLogger(__name__)

FINETUNE_AVAILABLE = False
_finetune_import_error = None
try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import torch
    FINETUNE_AVAILABLE = True
except Exception as e:
    _finetune_import_error = e
    FastLanguageModel = None
    SFTTrainer = None
    SFTConfig = None
    Dataset = None
    torch = None

# Map Ollama model names to HuggingFace/unsloth model identifiers
MODEL_MAP = {
    "nemotron-3-nano": "unsloth/Nemotron-3-Nano-30B-A3B",
    "llama3": "unsloth/llama-3-8b-bnb-4bit",
    "llama3.1": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "llama3.2": "unsloth/Llama-3.2-3B-bnb-4bit",
    "mistral": "unsloth/mistral-7b-v0.3-bnb-4bit",
    "gemma2": "unsloth/gemma-2-9b-bnb-4bit",
    "phi3": "unsloth/Phi-3.5-mini-instruct",
    "qwen2.5": "unsloth/Qwen2.5-7B-bnb-4bit",
}

# Nemotron currently tends to be unstable with 4-bit initialization on recent
# transformers/bitsandbytes combos, so prefer 16-bit loading first.
PREFER_16BIT_MODELS = {"nemotron-3-nano"}


class TrainingDataGenerator:
    """Converts scan results into SFT training data."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def generate_refusal(self, attack_payload: str) -> str:
        """Ask the LLM to generate an ideal safety refusal for an attack."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a safety alignment expert. Given an adversarial prompt, "
                    "write the ideal safe refusal response. Be firm but polite. "
                    "Explain that you cannot comply and briefly state why. "
                    "Reply with ONLY the refusal text, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Write a safe refusal for this adversarial prompt:\n\n{attack_payload}",
            },
        ]
        return await self.llm.chat(messages, temperature=0.3)

    async def generate_training_data(self, scan_result: ScanResult) -> list[dict]:
        """Generate SFT training pairs from bypassed attacks in scan results."""
        bypassed = [r for r in scan_result.results if r.bypassed]
        training_data = []

        for result in bypassed:
            refusal = await self.generate_refusal(result.attack.payload)
            training_data.append({
                "conversations": [
                    {"role": "user", "content": result.attack.payload},
                    {"role": "assistant", "content": refusal},
                ],
            })

        return training_data


class ModelHardener:
    """Runs LoRA fine-tuning and exports to GGUF for Ollama."""

    def __init__(
        self,
        ollama_model: str,
        output_dir: str = "./hardened_model",
        lora_rank: int = 16,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 2,
        quantization: str = "q4_k_m",
    ):
        if not FINETUNE_AVAILABLE:
            raise RuntimeError(
                "Fine-tuning dependencies not installed. "
                "Install with: pip install -e \".[hardening]\""
            )

        self.ollama_model = ollama_model
        self.output_dir = Path(output_dir)
        self.lora_rank = lora_rank
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.quantization = quantization
        self.model = None
        self.tokenizer = None

    def _resolve_model_name(self, ollama_model: str) -> str:
        """Map Ollama model name to HuggingFace/unsloth identifier."""
        base_name = ollama_model.split(":")[0]
        if base_name in MODEL_MAP:
            return MODEL_MAP[base_name]
        return f"unsloth/{ollama_model}"

    def _base_model_name(self) -> str:
        """Return the Ollama model name without tag."""
        return self.ollama_model.split(":")[0].lower()

    def _preferred_dtype(self):
        """Pick a practical dtype for non-4bit loading when available."""
        if torch is None or not torch.cuda.is_available():
            return None
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _build_load_attempts(self) -> list[dict]:
        """Build model loading attempts from most to least preferred."""
        preferred_16bit = self._base_model_name() in PREFER_16BIT_MODELS
        dtype = self._preferred_dtype()

        if preferred_16bit:
            return [
                {"load_in_4bit": False, "dtype": dtype, "reason": "preferred_16bit"},
            ]

        return [
            {"load_in_4bit": True, "dtype": None, "reason": "default_4bit"},
            {"load_in_4bit": False, "dtype": dtype, "reason": "fallback_16bit"},
        ]

    @staticmethod
    def _is_known_4bit_init_error(exc: Exception) -> bool:
        """Detect the 4-bit quantized weight init issue seen on some models."""
        msg = str(exc).lower()
        return "result type float can't be cast to the desired output type byte" in msg

    def load_model(self):
        """Load the base model and apply LoRA, with robust loading fallbacks."""
        hf_name = self._resolve_model_name(self.ollama_model)
        logger.info("Loading model: %s", hf_name)

        attempts = self._build_load_attempts()
        load_error = None
        for idx, attempt in enumerate(attempts, start=1):
            kwargs = {
                "model_name": hf_name,
                "max_seq_length": 2048,
                "load_in_4bit": attempt["load_in_4bit"],
                "trust_remote_code": True,
            }
            if attempt["dtype"] is not None:
                kwargs["dtype"] = attempt["dtype"]

            try:
                logger.info(
                    "Loading attempt %s/%s (load_in_4bit=%s, reason=%s)",
                    idx,
                    len(attempts),
                    attempt["load_in_4bit"],
                    attempt["reason"],
                )
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(**kwargs)
                break
            except Exception as exc:
                load_error = exc
                should_warn_known = attempt["load_in_4bit"] and self._is_known_4bit_init_error(exc)
                logger.warning(
                    "Model load attempt %s/%s failed%s: %s",
                    idx,
                    len(attempts),
                    " (known 4-bit init issue)" if should_warn_known else "",
                    exc,
                )
                if idx == len(attempts):
                    raise RuntimeError(
                        f"Failed to load model '{hf_name}' for hardening after {len(attempts)} attempt(s)."
                    ) from load_error

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=self.lora_rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "in_proj", "out_proj",
            ],
            lora_alpha=self.lora_rank,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    def _format_training_data(self, data: list[dict]) -> "Dataset":
        """Apply chat template to training data and return HF Dataset."""
        formatted = []
        for item in data:
            text = self.tokenizer.apply_chat_template(
                item["conversations"], tokenize=False, add_generation_prompt=False,
            )
            formatted.append({"text": text})
        return Dataset.from_list(formatted)

    def train(self, training_data: list[dict]) -> dict:
        """Run SFT training and export to GGUF."""
        if self.model is None:
            self.load_model()

        dataset = self._format_training_data(training_data)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=str(self.output_dir),
                per_device_train_batch_size=self.batch_size,
                num_train_epochs=self.num_epochs,
                learning_rate=self.learning_rate,
                optim="adamw_8bit",
                logging_steps=1,
                save_strategy="no",
                report_to="none",
                dataset_text_field="text",
                max_seq_length=2048,
            ),
        )

        train_result = trainer.train()

        # Export to GGUF
        self.model.save_pretrained_gguf(
            str(self.output_dir),
            self.tokenizer,
            quantization_method=self.quantization,
        )

        return {
            "train_loss": train_result.training_loss,
            "train_samples": len(dataset),
        }

    def export_to_ollama(self) -> str:
        """Create Ollama model from exported GGUF."""
        hardened_name = f"{self.ollama_model}-hardened"

        # Find the GGUF file
        gguf_files = glob.glob(str(self.output_dir / "*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"No GGUF file found in {self.output_dir}")
        gguf_path = gguf_files[0]

        # Create Modelfile
        modelfile_path = self.output_dir / "Modelfile"
        modelfile_path.write_text(
            f"FROM {gguf_path}\n"
            f"PARAMETER temperature 0.7\n"
            f"SYSTEM You are a helpful assistant.\n"
        )

        # Import into Ollama
        subprocess.run(
            ["ollama", "create", hardened_name, "-f", str(modelfile_path)],
            check=True,
            capture_output=True,
        )

        return hardened_name


async def run_hardening_pipeline(
    scan_result: ScanResult,
    llm: LLMProvider,
    ollama_model: str,
    output_dir: str,
    console: Console,
    lora_rank: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    quantization: str = "q4_k_m",
) -> HardeningResult:
    """Top-level orchestrator for the hardening pipeline."""
    bypassed = [r for r in scan_result.results if r.bypassed]
    if not bypassed:
        console.print("[yellow]No bypassed attacks found — skipping hardening.[/]")
        return HardeningResult(status="skipped")

    console.print(f"\n[bold cyan]Hardening pipeline — {len(bypassed)} bypassed attacks[/]")

    # 1. Generate training data
    console.print("[cyan]Generating refusal training data...[/]")
    generator = TrainingDataGenerator(llm)
    training_data = await generator.generate_training_data(scan_result)
    console.print(f"  Generated {len(training_data)} training examples")

    # 2. Fine-tune
    console.print("[cyan]Fine-tuning with LoRA...[/]")
    hardener = ModelHardener(
        ollama_model=ollama_model,
        output_dir=output_dir,
        lora_rank=lora_rank,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        quantization=quantization,
    )
    train_stats = hardener.train(training_data)
    console.print(f"  Training loss: {train_stats['train_loss']:.4f}")

    # 3. Export to Ollama
    console.print("[cyan]Exporting to Ollama...[/]")
    hardened_model = hardener.export_to_ollama()
    console.print(f"  Created model: [bold]{hardened_model}[/]")

    return HardeningResult(
        status="completed",
        hardened_model=hardened_model,
        training_examples=train_stats["train_samples"],
        train_loss=train_stats["train_loss"],
        original_score=scan_result.vulnerability_score,
    )

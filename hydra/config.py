from pydantic_settings import BaseSettings


class HydraConfig(BaseSettings):
    ollama_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "nemotron-3-nano"
    default_temperature: float = 0.7
    max_tokens: int = 4096
    scan_rounds: int = 5
    attacks_per_agent: int = 3
    mutations_per_attack: int = 2
    guardrails_enabled: bool = False
    guardrails_model: str = ""

    # Hardening (LoRA fine-tuning) settings
    hardening_output_dir: str = "./hardened_model"
    hardening_lora_rank: int = 16
    hardening_epochs: int = 3
    hardening_learning_rate: float = 2e-4
    hardening_batch_size: int = 2
    hardening_quantization: str = "q4_k_m"

    model_config = {"env_file": ".env", "env_prefix": "HYDRA_", "extra": "ignore"}

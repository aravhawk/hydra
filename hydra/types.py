from pydantic import BaseModel, Field
from enum import Enum
import uuid


class AttackType(str, Enum):
    INJECTION = "injection"
    JAILBREAK = "jailbreak"
    EXFILTRATION = "exfiltration"
    MUTATION = "mutation"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class Attack(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    attack_type: AttackType
    agent: str
    payload: str
    technique: str = ""
    parent_id: str | None = None
    generation: int = 0


class AttackResult(BaseModel):
    attack: Attack
    target_response: str
    bypassed: bool  # True = attack succeeded (bad for the target)
    severity: Severity = Severity.NONE
    categories: list[str] = Field(default_factory=list)
    guardrails_blocked: bool = False
    guardrails_response: str = ""
    reached_target: bool = True


class Defense(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    trigger_attack_id: str
    colang_code: str
    config_yaml: str


class ScanResult(BaseModel):
    target_name: str
    rounds: int = 0
    total_attacks: int = 0
    bypassed: int = 0
    blocked: int = 0
    vulnerability_score: float = 0.0
    results: list[AttackResult] = Field(default_factory=list)
    defenses: list[Defense] = Field(default_factory=list)
    duration_seconds: float = 0.0
    guardrails_enabled: bool = False
    guardrails_blocked_count: int = 0
    guardrails_bypassed_count: int = 0


class HardeningResult(BaseModel):
    status: str
    hardened_model: str = ""
    training_examples: int = 0
    train_loss: float = 0.0
    original_score: float = 0.0
    hardened_score: float = 0.0
    improvement: float = 0.0

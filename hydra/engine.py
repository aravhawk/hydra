import time
import logging
from rich.console import Console
from rich.progress import Progress
from hydra.types import ScanResult, AttackResult, Severity
from hydra.llm import LLMProvider
from hydra.agents.injector import Injector
from hydra.agents.jailbreaker import Jailbreaker
from hydra.agents.exfiltrator import Exfiltrator
from hydra.agents.mutator import Mutator
from hydra.agents.classifier import Classifier
from hydra.agents.defender import Defender

logger = logging.getLogger(__name__)
console = Console()


class HydraEngine:
    def __init__(self, llm: LLMProvider, target,
                 guardrails_enabled: bool = False,
                 guardrails_model: str = ""):
        self.llm = llm
        self.target = target
        self.red_agents = [Injector(llm), Jailbreaker(llm), Exfiltrator(llm)]
        self.mutator = Mutator(llm)
        self.classifier = Classifier(llm)
        self.defender = Defender(llm)
        self._guardrails = None
        if guardrails_enabled:
            self._guardrails = self._init_guardrails(guardrails_model)

    def _init_guardrails(self, guardrails_model: str):
        """Initialize GuardrailsManager, returning None if unavailable."""
        try:
            from hydra.guardrails import GuardrailsManager, GUARDRAILS_AVAILABLE
            if not GUARDRAILS_AVAILABLE:
                console.print(
                    "[yellow]Warning:[/] nemoguardrails not installed. "
                    "Guardrails disabled. Install with: pip install nemoguardrails[openai]"
                )
                return None

            model = guardrails_model or self.llm.model
            return GuardrailsManager(
                model=model,
                base_url=str(self.llm.client.base_url),
            )
        except Exception as e:
            logger.error("Failed to initialize guardrails: %s", e)
            console.print(f"[yellow]Warning:[/] Guardrails init failed: {e}")
            return None

    async def run_scan(self, rounds: int = 5) -> ScanResult:
        scan = ScanResult(
            target_name=self.target.name,
            guardrails_enabled=self._guardrails is not None,
        )
        start = time.time()
        all_defenses = []

        with Progress(console=console) as progress:
            task = progress.add_task("[cyan]Scanning...", total=rounds)

            for round_num in range(rounds):
                console.print(f"\n[bold yellow]═══ Round {round_num + 1}/{rounds} ═══[/]")

                # 1. Generate attacks
                attacks = []
                for agent in self.red_agents:
                    context = {
                        "target_system_prompt": self.target.system_prompt,
                        "round": round_num,
                        "previous_results": scan.results,
                        "defenses": all_defenses,
                    }
                    new_attacks = await agent.run(context)
                    attacks.extend(new_attacks)
                    console.print(f"  [red]⚔ {agent.name}[/] generated {len(new_attacks)} attacks")

                # 2. Add mutations from previous round's successes
                if round_num > 0:
                    successful = [r for r in scan.results if r.bypassed]
                    if successful:
                        context["successful_attacks"] = successful
                        context["defenses"] = all_defenses
                        if self._guardrails is not None:
                            context["guardrails_active"] = True
                            context["guardrails_blocked"] = [
                                r for r in scan.results if r.guardrails_blocked
                            ]
                        mutations = await self.mutator.run(context)
                        attacks.extend(mutations)
                        console.print(f"  [red]🧬 Mutator[/] generated {len(mutations)} mutations")

                # 3. Send each attack to target and classify
                for attack in attacks:
                    # Test against guardrails first (if active and has defenses)
                    if self._guardrails is not None and self._guardrails.has_defenses:
                        gr_result = await self._guardrails.check(attack.payload)
                        if gr_result.blocked:
                            result = AttackResult(
                                attack=attack,
                                target_response="[BLOCKED BY GUARDRAILS]",
                                bypassed=False,
                                severity=Severity.NONE,
                                guardrails_blocked=True,
                                guardrails_response=gr_result.response,
                                reached_target=False,
                            )
                            scan.results.append(result)
                            console.print(
                                f"    [blue]GUARDRAIL ✓[/] "
                                f"[{attack.attack_type.value}] {attack.technique}"
                            )
                            continue

                    response = await self.target.send(attack.payload)
                    result = await self.classifier.classify(attack, response)
                    scan.results.append(result)

                    status = "[red]BYPASSED ✗[/]" if result.bypassed else "[green]BLOCKED ✓[/]"
                    console.print(f"    {status} [{attack.attack_type.value}] {attack.technique}")

                    # 4. Generate defense for successful attacks
                    if result.bypassed:
                        defense = await self.defender.defend(result)
                        if defense:
                            all_defenses.append(defense)
                            scan.defenses.append(defense)
                            console.print(f"    [blue]🛡 Defense generated:[/] {defense.name}")

                            if self._guardrails is not None:
                                self._guardrails.add_defense(defense)
                                console.print(
                                    f"    [blue]🛡 Deployed to guardrails[/] "
                                    f"({self._guardrails.defense_count} total)"
                                )

                progress.advance(task)

        # Compute final stats
        scan.rounds = rounds
        scan.total_attacks = len(scan.results)
        scan.bypassed = sum(1 for r in scan.results if r.bypassed)
        scan.blocked = scan.total_attacks - scan.bypassed
        scan.vulnerability_score = scan.bypassed / max(scan.total_attacks, 1)
        scan.duration_seconds = time.time() - start

        if self._guardrails is not None:
            scan.guardrails_blocked_count = sum(
                1 for r in scan.results if r.guardrails_blocked
            )
            scan.guardrails_bypassed_count = sum(
                1 for r in scan.results if r.reached_target and not r.guardrails_blocked
            )

        return scan

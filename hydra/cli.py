import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from hydra.config import HydraConfig
from hydra.llm import LLMProvider
from hydra.target import Target
from hydra.engine import HydraEngine

app = typer.Typer(name="hydra", help="AI Security Platform — continuous adversarial testing",
                  invoke_without_command=True)
console = Console()


@app.callback(invoke_without_command=True)
def scan(
    model: str = typer.Option(None, help="Target model to attack (defaults to config LLM model)"),
    rounds: int = typer.Option(5, help="Number of adversarial rounds"),
    system_prompt: str = typer.Option(
        "You are a helpful assistant.",
        help="Target's system prompt (for targeted attacks)",
    ),
    guardrails: bool = typer.Option(
        False, "--guardrails",
        help="Enable live NeMo Guardrails defense testing",
    ),
    export_defenses: str = typer.Option(
        None, "--export-defenses",
        help="Export generated defenses to a directory",
    ),
    harden: bool = typer.Option(
        False, "--harden",
        help="Run LoRA fine-tuning pipeline after scan to harden the model",
    ),
    verify: bool = typer.Option(
        False, "--verify",
        help="Re-scan the hardened model to measure improvement (requires --harden)",
    ),
):
    """Run an adversarial security scan against a model on Ollama."""
    config = HydraConfig()
    llm = LLMProvider(base_url=config.ollama_base_url, model=config.llm_model)
    target_model = model or config.llm_model

    target = Target(
        name=target_model,
        base_url=config.ollama_base_url,
        model=target_model,
        system_prompt=system_prompt,
    )

    use_guardrails = guardrails or config.guardrails_enabled

    console.print(Panel(
        f"[bold]Target:[/] {target_model}\n"
        f"[bold]Rounds:[/] {rounds}\n"
        f"[bold]Attacker LLM:[/] {config.llm_model}\n"
        f"[bold]Ollama:[/] {config.ollama_base_url}\n"
        f"[bold]Guardrails:[/] {'enabled' if use_guardrails else 'disabled'}",
        title="Hydra Security Scan",
    ))

    engine = HydraEngine(
        llm=llm, target=target,
        guardrails_enabled=use_guardrails,
        guardrails_model=config.guardrails_model,
    )
    result = asyncio.run(engine.run_scan(rounds=rounds))
    _print_report(result)

    # Defense export
    if export_defenses and result.defenses:
        from hydra.export import DefenseExporter
        exporter = DefenseExporter()
        out = exporter.export(
            result.defenses, export_defenses, target_model, config.ollama_base_url,
        )
        if out:
            console.print(Panel(
                f"[bold]Directory:[/] {out}\n"
                f"[bold]Defenses:[/] {len(result.defenses)}\n"
                f"[bold]Files:[/] config.yml, defenses.co, manifest.json",
                title="Defenses Exported",
            ))
    elif export_defenses and not result.defenses:
        console.print("[yellow]No defenses generated — nothing to export.[/]")

    # Hardening
    harden_result = None
    if harden:
        from hydra.hardening import FINETUNE_AVAILABLE, run_hardening_pipeline, _finetune_import_error
        if not FINETUNE_AVAILABLE:
            msg = "Fine-tuning dependencies not installed."
            if _finetune_import_error:
                msg += f" Error: {_finetune_import_error}"
            console.print(f"[yellow]Warning:[/] {msg}")
            console.print("Install with: pip install -e \".[hardening]\"")

        else:
            harden_result = asyncio.run(run_hardening_pipeline(
                scan_result=result,
                llm=llm,
                ollama_model=target_model,
                output_dir=config.hardening_output_dir,
                console=console,
                lora_rank=config.hardening_lora_rank,
                num_epochs=config.hardening_epochs,
                learning_rate=config.hardening_learning_rate,
                batch_size=config.hardening_batch_size,
                quantization=config.hardening_quantization,
            ))
            if harden_result and harden_result.status == "completed":
                _print_hardening_report(harden_result)

    # Verification
    if verify and harden_result and harden_result.status == "completed":
        console.print("\n[bold cyan]Verification — re-scanning hardened model...[/]")
        hardened_target = Target(
            name=harden_result.hardened_model,
            base_url=config.ollama_base_url,
            model=harden_result.hardened_model,
            system_prompt=system_prompt,
        )
        hardened_engine = HydraEngine(llm=llm, target=hardened_target)
        hardened_result = asyncio.run(hardened_engine.run_scan(rounds=rounds))
        _print_report(hardened_result)
        _print_verification(result, hardened_result)
    elif verify and not harden:
        console.print("[yellow]--verify requires --harden.[/]")


def _print_report(result):
    console.print(f"\n[bold]{'=' * 60}[/]")

    lines = [
        f"[bold]Vulnerability Score:[/] {result.vulnerability_score:.0%}",
        f"[bold]Total Attacks:[/] {result.total_attacks}",
        f"[bold]Bypassed:[/] [red]{result.bypassed}[/]",
        f"[bold]Blocked:[/] [green]{result.blocked}[/]",
    ]
    if result.guardrails_enabled:
        lines.append(f"[bold]Blocked by Guardrails:[/] [blue]{result.guardrails_blocked_count}[/]")
        lines.append(f"[bold]Bypassed Guardrails:[/] [yellow]{result.guardrails_bypassed_count}[/]")
    lines.append(f"[bold]Defenses Generated:[/] {len(result.defenses)}")
    lines.append(f"[bold]Duration:[/] {result.duration_seconds:.1f}s")

    console.print(Panel("\n".join(lines), title="Scan Results"))

    if result.defenses:
        console.print("\n[bold blue]Generated NeMo Guardrails Defenses:[/]")
        for d in result.defenses:
            console.print(f"\n[bold]{d.name}[/]")
            console.print(f"```colang\n{d.colang_code}\n```")

    bypassed = [r for r in result.results if r.bypassed]
    if bypassed:
        table = Table(title="Top Successful Attacks")
        table.add_column("Type", style="red")
        table.add_column("Technique")
        table.add_column("Severity")
        table.add_column("Payload", max_width=60)
        for r in bypassed[:10]:
            table.add_row(
                r.attack.attack_type.value,
                r.attack.technique,
                r.severity.value,
                r.attack.payload[:60] + "..." if len(r.attack.payload) > 60 else r.attack.payload,
            )
        console.print(table)


def _print_hardening_report(harden_result):
    console.print(Panel(
        f"[bold]Model:[/] {harden_result.hardened_model}\n"
        f"[bold]Training Examples:[/] {harden_result.training_examples}\n"
        f"[bold]Final Loss:[/] {harden_result.train_loss:.4f}\n"
        f"[bold]Status:[/] [green]{harden_result.status}[/]",
        title="Hardening Complete",
    ))


def _print_verification(original, hardened):
    original_score = original.vulnerability_score
    hardened_score = hardened.vulnerability_score
    improvement = original_score - hardened_score

    color = "green" if improvement > 0 else ("yellow" if improvement == 0 else "red")
    console.print(Panel(
        f"[bold]Original Score:[/] {original_score:.0%}\n"
        f"[bold]Hardened Score:[/] {hardened_score:.0%}\n"
        f"[bold]Improvement:[/] [{color}]{improvement:+.0%}[/{color}]",
        title="Verification Results",
    ))


if __name__ == "__main__":
    app()

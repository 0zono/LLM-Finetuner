from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from src.core.config import PipelineConfig
from src.core.llm_client import LocalLLMClient
from src.curation.llm_judge import curate_records
from src.export.jsonl_exporter import export_run
from src.formatting.chat_formatter import format_records
from src.generation.llm_generator import generate_examples
from src.ingestion.seed_loader import load_seed_records
from src.preprocessing.cleaner import clean_records
from src.validation.schema_validator import validate_records
from src.validation.evaluator import evaluate_validator


class PipelineOrchestrator:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        needs_llm = config.enable_generation or (
            config.enable_curation and config.curation_mode == "llm"
        )
        if needs_llm and not config.llm.enabled:
            raise ValueError(
                "A configuração exige LM, mas llm.enabled está desabilitado"
            )
        self.client = LocalLLMClient(config.llm) if needs_llm else None
        self.stats: dict[str, Any] = {}

    def _measure(self, name: str, operation: Callable[[], Any]) -> Any:
        started = perf_counter()
        result = operation()
        elapsed = perf_counter() - started
        self.stats.setdefault(name, {})["duration_seconds"] = round(elapsed, 6)
        return result

    def run(self) -> dict[str, Any]:
        records = self._measure(
            "ingestion",
            lambda: load_seed_records(self.config.seed_path, self.config.task_type),
        )
        self.stats["ingestion"]["output"] = len(records)

        cleaned, rejected_cleaning = self._measure(
            "preprocessing", lambda: clean_records(records)
        )
        self.stats["preprocessing"].update(
            input=len(records), output=len(cleaned), rejected=len(rejected_cleaning)
        )

        generated, rejected_generation = self._measure(
            "generation",
            lambda: generate_examples(
                cleaned,
                enabled=self.config.enable_generation,
                variations_per_seed=self.config.variations_per_seed,
                client=self.client,
            ),
        )
        self.stats["generation"].update(
            input=len(cleaned), output=len(generated), rejected=len(rejected_generation)
        )

        formatted, rejected_formatting = self._measure(
            "formatting",
            lambda: format_records(generated, self.config.system_prompt),
        )
        self.stats["formatting"].update(
            input=len(generated), output=len(formatted), rejected=len(rejected_formatting)
        )

        curated, rejected_curation = self._measure(
            "curation",
            lambda: curate_records(
                formatted,
                enabled=self.config.enable_curation,
                mode=self.config.curation_mode,
                minimum_score=self.config.minimum_curation_score,
                client=self.client,
            ),
        )
        self.stats["curation"].update(
            input=len(formatted), output=len(curated), rejected=len(rejected_curation)
        )

        valid, rejected_validation = self._measure(
            "validation", lambda: validate_records(curated)
        )
        self.stats["validation"].update(
            input=len(curated), output=len(valid), rejected=len(rejected_validation)
        )

        evaluation = None
        if self.config.enable_evaluation:
            evaluation = self._measure(
                "evaluation",
                lambda: evaluate_validator(
                    valid, f"{self.config.reports_dir}/validator_evaluation.json"
                ),
            )
            self.stats["evaluation"]["examples"] = evaluation["examples"]

        invalid = (
            rejected_cleaning
            + rejected_generation
            + rejected_formatting
            + rejected_curation
            + rejected_validation
        )
        manifest = self._measure(
            "export",
            lambda: export_run(
                valid,
                invalid,
                self.config,
                self.stats,
                self.client.usage if self.client else None,
            ),
        )
        self.stats["export"].update(valid=len(valid), invalid=len(invalid))
        return {
            "total_records": len(records),
            "cleaned_records": len(cleaned),
            "generated_records": len(generated),
            "formatted_records": len(formatted),
            "curated_records": len(curated),
            "valid_records": len(valid),
            "invalid_records": len(invalid),
            "evaluation": evaluation,
            "manifest": manifest,
        }

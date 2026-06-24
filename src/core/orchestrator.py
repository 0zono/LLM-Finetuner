from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from src.core.config import PipelineConfig
from src.core.llm_client import LocalLLMClient
from src.core.models import TaskType
from src.core.tool_registry import ToolRegistry
from src.curation.llm_judge import curate_records
from src.export.jsonl_exporter import assign_seed_splits, export_run
from src.formatting.chat_formatter import format_records
from src.generation.llm_generator import generate_examples
from src.ingestion.seed_loader import load_seed_records
from src.preprocessing.cleaner import clean_records
from src.validation.schema_validator import validate_records
from src.validation.evaluator import evaluate_validator


class PipelineOrchestrator:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.tool_registry = (
            ToolRegistry.from_file(config.tools_file)
            if config.task_type == TaskType.TOOL_CALLING and config.tools_file
            else None
        )
        if self.tool_registry and self.tool_registry.domain != config.domain:
            raise ValueError(
                "O domínio do config não corresponde ao domínio do registro de ferramentas: "
                f"{config.domain} != {self.tool_registry.domain}"
            )
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

        seed_splits = self._measure(
            "splitting", lambda: assign_seed_splits(cleaned, self.config)
        )
        self.stats["splitting"].update(
            train=len(seed_splits["train"]),
            validation=len(seed_splits["validation"]),
            test=len(seed_splits["test"]),
        )

        augmented_train, rejected_generation = self._measure(
            "generation",
            lambda: generate_examples(
                seed_splits["train"],
                enabled=self.config.enable_generation,
                variations_per_seed=self.config.variations_per_seed,
                client=self.client,
            ),
        )
        train_variations = [
            record for record in augmented_train if record.meta.get("generated")
        ]
        combined = cleaned + train_variations
        self.stats["generation"].update(
            input=len(seed_splits["train"]),
            added=len(train_variations),
            output=len(combined),
            rejected=len(rejected_generation),
        )

        prepared, rejected_post_generation = self._measure(
            "post_generation_cleaning", lambda: clean_records(combined)
        )
        self.stats["post_generation_cleaning"].update(
            input=len(combined),
            output=len(prepared),
            rejected=len(rejected_post_generation),
        )
        prepared_variations = sum(
            bool(record.meta.get("generated")) for record in prepared
        )
        self.stats["generation"]["accepted_variations"] = prepared_variations

        formatted, rejected_formatting = self._measure(
            "formatting",
            lambda: format_records(prepared, self.config.system_prompt),
        )
        self.stats["formatting"].update(
            input=len(prepared), output=len(formatted), rejected=len(rejected_formatting)
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
            "validation", lambda: validate_records(curated, self.tool_registry)
        )
        self.stats["validation"].update(
            input=len(curated), output=len(valid), rejected=len(rejected_validation)
        )

        evaluation = None
        if self.config.enable_evaluation:
            evaluation = self._measure(
                "evaluation",
                lambda: evaluate_validator(
                    valid,
                    f"{self.config.reports_dir}/validator_evaluation.json",
                    self.tool_registry,
                ),
            )
            self.stats["evaluation"]["examples"] = evaluation["examples"]

        invalid = (
            rejected_cleaning
            + rejected_generation
            + rejected_post_generation
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
            "generated_records": len(prepared),
            "generated_variations": prepared_variations,
            "formatted_records": len(formatted),
            "curated_records": len(curated),
            "valid_records": len(valid),
            "invalid_records": len(invalid),
            "evaluation": evaluation,
            "manifest": manifest,
        }

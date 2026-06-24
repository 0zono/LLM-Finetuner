from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.models import TaskType


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    base_url: str = "http://localhost:1234/v1"
    model: str = "local-model"
    api_key_env: str = "LOCAL_LLM_API_KEY"
    temperature: float = Field(default=0.2, ge=0, le=2)
    timeout_seconds: int = Field(default=120, gt=0)
    max_retries: int = Field(default=3, ge=0)
    cache_dir: str = ".cache/llm"

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "local")


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1
    seed: int = 42


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_type: TaskType = TaskType.TOOL_CALLING
    domain: str = "generic"
    tools_file: str | None = None
    seed_path: str = "data/seeds/examples.json"
    output_dir: str = "data/output/latest"
    reports_dir: str = "reports/latest"
    enable_generation: bool = False
    variations_per_seed: int = Field(default=1, ge=0)
    enable_curation: bool = True
    enable_evaluation: bool = True
    curation_mode: Literal["heuristic", "llm"] = "heuristic"
    minimum_curation_score: float = Field(default=0.7, ge=0, le=1)
    system_prompt: str = "Você é um assistente que pode operar as ferramentas fornecidas."
    splits: SplitConfig = Field(default_factory=SplitConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @model_validator(mode="after")
    def validate_domain_configuration(self) -> "PipelineConfig":
        if self.task_type == TaskType.TOOL_CALLING and not self.tools_file:
            raise ValueError("task_type tool_calling exige tools_file")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuração não encontrada: {path}")
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        return cls.model_validate(data)

    def validate_split_sum(self) -> None:
        total = self.splits.train + self.splits.validation + self.splits.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError("As proporções de split devem somar 1.0")

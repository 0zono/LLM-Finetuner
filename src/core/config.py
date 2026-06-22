from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

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
    seed_path: str = "data/seeds/suporte_tecnico.json"
    output_dir: str = "data/output/latest"
    reports_dir: str = "reports/latest"
    enable_generation: bool = False
    variations_per_seed: int = Field(default=1, ge=0)
    enable_curation: bool = True
    enable_evaluation: bool = True
    curation_mode: Literal["heuristic", "llm"] = "heuristic"
    minimum_curation_score: float = Field(default=0.7, ge=0, le=1)
    system_prompt: str = (
        "Você é um agente responsável por operar ferramentas de suporte técnico. "
        "Quando necessário, converta a solicitação em uma chamada estruturada."
    )
    splits: SplitConfig = Field(default_factory=SplitConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
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

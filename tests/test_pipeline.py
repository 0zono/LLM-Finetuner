import json
from pathlib import Path

from src.core.config import PipelineConfig, SplitConfig
from src.core.orchestrator import PipelineOrchestrator
from src.ingestion.seed_loader import load_seed_records


SUPPORT_TOOLS = "domains/suporte_tecnico/tools.json"


def create_seeds(path: Path) -> None:
    seeds = [
        {
            "text": f"Consulte o chamado {index}.",
            "tool": "consultar_chamado",
            "arguments": {"chamado_id": index},
        }
        for index in range(1, 7)
    ]
    path.write_text(json.dumps(seeds, ensure_ascii=False), encoding="utf-8")


def test_ids_are_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "seeds.json"
    create_seeds(source)
    first = load_seed_records(source)
    second = load_seed_records(source)
    assert [record.id for record in first] == [record.id for record in second]


def test_pipeline_exports_reproducible_splits(tmp_path: Path) -> None:
    source = tmp_path / "seeds.json"
    create_seeds(source)
    output = tmp_path / "output"
    reports = tmp_path / "reports"
    config = PipelineConfig(
        seed_path=str(source),
        output_dir=str(output),
        reports_dir=str(reports),
        tools_file=SUPPORT_TOOLS,
        domain="suporte_tecnico",
        enable_generation=False,
        enable_curation=True,
        enable_evaluation=True,
        splits=SplitConfig(train=0.6, validation=0.2, test=0.2, seed=7),
    )
    result = PipelineOrchestrator(config).run()
    assert result["valid_records"] == 6
    assert result["invalid_records"] == 0
    assert (output / "train.jsonl").exists()
    assert (output / "validation.jsonl").exists()
    assert (output / "test.jsonl").exists()
    assert (output / "invalid.jsonl").exists()
    assert (reports / "manifest.json").exists()
    assert result["evaluation"]["f1_valid"] == 1.0

    first_manifest = json.loads((reports / "manifest.json").read_text(encoding="utf-8"))
    PipelineOrchestrator(config).run()
    second_manifest = json.loads((reports / "manifest.json").read_text(encoding="utf-8"))
    for split in ("train", "validation", "test"):
        assert (
            first_manifest["artifacts"][split]["sha256"]
            == second_manifest["artifacts"][split]["sha256"]
        )


def test_invalid_seed_reaches_invalid_export(tmp_path: Path) -> None:
    source = tmp_path / "seeds.json"
    source.write_text(
        json.dumps(
            [
                {
                    "text": "Use uma ferramenta inexistente.",
                    "tool": "inexistente",
                    "arguments": {"x": 1},
                },
                {
                    "text": "Consulte o chamado 1.",
                    "tool": "consultar_chamado",
                    "arguments": {"chamado_id": 1},
                },
                {
                    "text": "Consulte o chamado 2.",
                    "tool": "consultar_chamado",
                    "arguments": {"chamado_id": 2},
                },
                {
                    "text": "Consulte o chamado 3.",
                    "tool": "consultar_chamado",
                    "arguments": {"chamado_id": 3},
                },
            ]
        ),
        encoding="utf-8",
    )
    config = PipelineConfig(
        seed_path=str(source),
        output_dir=str(tmp_path / "output"),
        reports_dir=str(tmp_path / "reports"),
        tools_file=SUPPORT_TOOLS,
        domain="suporte_tecnico",
        enable_generation=False,
        enable_evaluation=False,
    )
    result = PipelineOrchestrator(config).run()
    assert result["invalid_records"] == 1
    invalid_text = (tmp_path / "output" / "invalid.jsonl").read_text(encoding="utf-8")
    assert "UNKNOWN_TOOL" in invalid_text


def test_second_domain_runs_without_core_changes(tmp_path: Path) -> None:
    config = PipelineConfig(
        domain="biblioteca",
        tools_file="domains/biblioteca/tools.json",
        seed_path="data/seeds/biblioteca.json",
        output_dir=str(tmp_path / "biblioteca-output"),
        reports_dir=str(tmp_path / "biblioteca-reports"),
        enable_generation=False,
        enable_evaluation=True,
    )
    result = PipelineOrchestrator(config).run()
    assert result["total_records"] == 8
    assert result["valid_records"] == 8
    assert result["invalid_records"] == 0


class FakePipelineClient:
    def __init__(self):
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def chat_json(self, messages, **kwargs):
        self.usage["requests"] += 1
        request = json.loads(messages[1]["content"])
        chamado_id = request["arguments"]["chamado_id"]
        return {
            "variations": [
                f"Verifique o chamado {chamado_id}, por favor.",
                f"Consulte a situação do chamado {chamado_id}.",
                f"Quero informações do chamado {chamado_id}.",
            ]
        }


def test_generation_is_applied_only_to_training_split(tmp_path: Path) -> None:
    source = tmp_path / "seeds.json"
    create_seeds(source)
    config = PipelineConfig(
        seed_path=str(source),
        output_dir=str(tmp_path / "output"),
        reports_dir=str(tmp_path / "reports"),
        tools_file=SUPPORT_TOOLS,
        domain="suporte_tecnico",
        enable_generation=True,
        variations_per_seed=1,
        enable_curation=True,
        enable_evaluation=False,
        splits=SplitConfig(train=0.6, validation=0.2, test=0.2, seed=7),
        llm={"enabled": True},
    )
    orchestrator = PipelineOrchestrator(config)
    orchestrator.client = FakePipelineClient()
    result = orchestrator.run()

    assert result["generated_variations"] == 4
    manifest = result["manifest"]
    assert manifest["artifacts"]["train"]["records"] == 8
    assert manifest["artifacts"]["validation"]["records"] == 1
    assert manifest["artifacts"]["test"]["records"] == 1

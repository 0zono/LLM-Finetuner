import json
from pathlib import Path

from src.core.config import PipelineConfig, SplitConfig
from src.core.orchestrator import PipelineOrchestrator
from src.ingestion.seed_loader import load_seed_records


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
        enable_generation=False,
        enable_evaluation=False,
    )
    result = PipelineOrchestrator(config).run()
    assert result["invalid_records"] == 1
    invalid_text = (tmp_path / "output" / "invalid.jsonl").read_text(encoding="utf-8")
    assert "UNKNOWN_TOOL" in invalid_text

from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.config import PipelineConfig
from src.core.models import CanonicalRecord


def split_by_parent(
    records: list[CanonicalRecord], config: PipelineConfig
) -> dict[str, list[CanonicalRecord]]:
    config.validate_split_sum()
    groups: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        groups[record.parent_seed_id].append(record)
    keys = sorted(groups)
    random.Random(config.splits.seed).shuffle(keys)
    total = len(keys)
    active_splits = sum(
        value > 0
        for value in (config.splits.train, config.splits.validation, config.splits.test)
    )
    if total < active_splits:
        raise ValueError(
            f"São necessários ao menos {active_splits} grupos de origem para os splits configurados"
        )
    test_count = max(1, round(total * config.splits.test)) if config.splits.test else 0
    validation_count = (
        max(1, round(total * config.splits.validation))
        if config.splits.validation
        else 0
    )
    train_count = total - validation_count - test_count
    if config.splits.train and train_count < 1:
        raise ValueError("Não há grupos suficientes para produzir um conjunto de treino")
    train_end = train_count
    validation_end = train_end + validation_count
    partitions = {
        "train": keys[:train_end],
        "validation": keys[train_end:validation_end],
        "test": keys[validation_end:],
    }
    return {
        name: [record for key in parent_ids for record in groups[key]]
        for name, parent_ids in partitions.items()
    }


def export_run(
    valid: list[CanonicalRecord],
    invalid: list[CanonicalRecord],
    config: PipelineConfig,
    stage_stats: dict[str, Any],
    llm_usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    reports_dir = Path(config.reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    splits = split_by_parent(valid, config)
    artifacts: dict[str, dict[str, Any]] = {}
    for name, records in splits.items():
        path = output_dir / f"{name}.jsonl"
        write_payload_jsonl(records, path)
        artifacts[name] = artifact_metadata(path, len(records))

    invalid_path = output_dir / "invalid.jsonl"
    write_record_jsonl(invalid, invalid_path)
    artifacts["invalid"] = artifact_metadata(invalid_path, len(invalid))

    error_counts = Counter(
        error.code for record in invalid for error in record.errors
    )
    manifest = {
        "schema_version": "1.0",
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "config": config.model_dump(mode="json"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "stage_stats": stage_stats,
        "llm_usage": llm_usage or {},
        "errors_by_code": dict(sorted(error_counts.items())),
        "artifacts": artifacts,
    }
    manifest_path = reports_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def write_payload_jsonl(records: list[CanonicalRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record.payload, ensure_ascii=False) + "\n")


def write_record_jsonl(records: list[CanonicalRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(record.model_dump_json() + "\n")


def artifact_metadata(path: Path, records: int) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": str(path),
        "records": records,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


# Compatibilidade com chamadas externas da primeira versão.
def export_jsonl(records: list[CanonicalRecord], output_path: str | Path) -> None:
    write_payload_jsonl(records, Path(output_path))

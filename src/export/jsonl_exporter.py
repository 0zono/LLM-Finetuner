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
    if records and all(record.meta.get("split") for record in records):
        splits = {"train": [], "validation": [], "test": []}
        parent_splits: dict[str, str] = {}
        for record in records:
            split = str(record.meta["split"])
            if split not in splits:
                raise ValueError(f"Split desconhecido no registro {record.id}: {split}")
            previous = parent_splits.setdefault(record.parent_seed_id, split)
            if previous != split:
                raise ValueError(
                    f"Vazamento: parent_seed_id {record.parent_seed_id} aparece em mais de um split"
                )
            splits[split].append(record)
        return splits

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


def assign_seed_splits(
    records: list[CanonicalRecord], config: PipelineConfig
) -> dict[str, list[CanonicalRecord]]:
    """Atribui splits antes do aumento, estratificando pela ferramenta/intenção."""
    config.validate_split_sum()
    grouped: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        grouped[record.tool or "__no_tool__"].append(record)

    splits = {"train": [], "validation": [], "test": []}
    proportions = {
        "train": config.splits.train,
        "validation": config.splits.validation,
        "test": config.splits.test,
    }
    for label in sorted(grouped):
        label_records = sorted(grouped[label], key=lambda item: item.id)
        label_seed = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
        random.Random(config.splits.seed + label_seed).shuffle(label_records)
        counts = _allocate_split_counts(len(label_records), proportions)
        offset = 0
        for split in ("train", "validation", "test"):
            selected = label_records[offset : offset + counts[split]]
            offset += counts[split]
            for record in selected:
                record.meta["split"] = split
                record.add_event("splitting", "assigned", {"split": split, "stratum": label})
            splits[split].extend(selected)
    return splits


def _allocate_split_counts(
    total: int, proportions: dict[str, float]
) -> dict[str, int]:
    counts = {name: 0 for name in proportions}
    active = [name for name, value in proportions.items() if value > 0]
    if total >= len(active):
        for name in active:
            counts[name] = 1
    else:
        priority = sorted(active, key=lambda name: (-proportions[name], name))
        for name in priority[:total]:
            counts[name] = 1

    remaining = total - sum(counts.values())
    for _ in range(remaining):
        chosen = max(
            active,
            key=lambda name: (proportions[name] * total - counts[name], proportions[name]),
        )
        counts[chosen] += 1
    return counts


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
